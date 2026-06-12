import os
import sys
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, accuracy_score)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '3D_GLAD'))
from model import DTIPredModel, GCNEncoder
from attention_data import build_datasets, DRUG_FEAT_DIM, PROTEIN_FEAT_DIM


def find_optimal_threshold(labels, preds):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def pretrain_and_save_encoder(train_ds, data_dir, log_dir, device, gae_epochs=30):
    encoder_path = os.path.join(log_dir, 'gcn_gae_encoder.pt')
    if os.path.exists(encoder_path):
        print(f"Encoder already exists at {encoder_path}, skipping GAE pre-training")
        return encoder_path

    print("Loading datasets for GAE pre-training...")
    _, _, _ = build_datasets(data_dir)

    print("Creating dummy model for GAE pre-training...")
    model = DTIPredModel(decoder_type='mlp', latent_dim=31).to(device)

    print(f"\nPhase 0: GAE Pre-training (GCN encoder)")
    print("=" * 60)

    drug_params = list(model.drug_encoder.parameters())
    protein_params = list(model.protein_encoder.parameters())

    drug_list = list(train_ds.drug_graphs.values())
    loader = DataLoader(drug_list, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(drug_params, lr=1e-3)

    for epoch in range(gae_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model.encode_drug(batch.x, batch.edge_index)
            loss = model.gae_recon_loss(z, batch.edge_index, batch.x.size(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drug_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  DrugGAE Epoch {epoch+1}/{gae_epochs} Loss: {total_loss/len(loader):.4f}")

    individual_pockets = []
    for pg in train_ds.protein_graphs.values():
        if isinstance(pg, Batch) and hasattr(pg, 'batch'):
            for i in range(pg.num_graphs):
                mask = (pg.batch == i)
                node_indices = mask.nonzero(as_tuple=True)[0]
                x = pg.x[mask]
                edge_mask = (pg.batch[pg.edge_index[0]] == i)
                local_edges = pg.edge_index[:, edge_mask]
                offset = node_indices[0].item()
                local_edges = local_edges - offset
                valid = (local_edges >= 0).all(dim=0) & (local_edges < x.size(0)).all(dim=0)
                local_edges = local_edges[:, valid]
                if x.size(0) > 0 and local_edges.size(1) > 0:
                    individual_pockets.append(Data(x=x, edge_index=local_edges))
        elif isinstance(pg, Data) and pg.x.size(1) == PROTEIN_FEAT_DIM:
            individual_pockets.append(pg)
    print(f"  Individual pocket graphs: {len(individual_pockets)}")
    loader = DataLoader(individual_pockets, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(protein_params, lr=1e-3)

    for epoch in range(gae_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            if batch.edge_index.size(1) == 0:
                continue
            optimizer.zero_grad()
            z = model.encode_protein(batch.x, batch.edge_index)
            loss = model.gae_recon_loss(z, batch.edge_index, batch.x.size(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(protein_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ProteinGAE Epoch {epoch+1}/{gae_epochs} Loss: "
                  f"{total_loss/max(len(loader),1):.4f}")

    encoder_state = {
        'drug_encoder': model.drug_encoder.state_dict(),
        'protein_encoder': model.protein_encoder.state_dict(),
    }
    torch.save(encoder_state, encoder_path)
    print(f"\nEncoder saved to {encoder_path}")
    print(f"  Drug encoder: {sum(p.numel() for p in model.drug_encoder.parameters()):,} params")
    print(f"  Protein encoder: {sum(p.numel() for p in model.protein_encoder.parameters()):,} params")
    return encoder_path


@torch.no_grad()
def evaluate(model, dataset, device, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []
    for idx in tqdm(range(len(dataset)), desc=desc, leave=False):
        pg, dg, label = dataset[idx]
        pg, dg = pg.to(device), dg.to(device)
        out = model(pg, dg)
        all_preds.append(out.item())
        all_labels.append(label)
        del pg, dg, out

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    opt_thresh = find_optimal_threshold(all_labels, all_preds)
    pred_binary = (all_preds >= opt_thresh).astype(int)
    auc_val = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    auprc = average_precision_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    f1 = f1_score(all_labels, pred_binary, zero_division=0)
    acc = accuracy_score(all_labels, pred_binary)
    return {'AUC': auc_val, 'AUPRC': auprc, 'F1': f1, 'Accuracy': acc, 'Threshold': opt_thresh}


def main():
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['RDKIT_ERROR_LEVEL'] = 'ERROR'
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default='mlp',
                        choices=['mlp', 'bilinear', 'ntn', 'nodebilinear'])
    parser.add_argument('--gae_epochs', type=int, default=30)
    parser.add_argument('--freeze_epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, 'InterpretableDTIP', 'data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    print(f"Decoder: {args.decoder} | Device: {device}")

    print("Loading datasets...")
    train_ds, test_ds, dev_ds = build_datasets(data_dir)
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}, Test: {len(test_ds)}")

    encoder_path = pretrain_and_save_encoder(train_ds, data_dir, log_dir, device,
                                              gae_epochs=args.gae_epochs)

    model = DTIPredModel(decoder_type=args.decoder, latent_dim=31).to(device)

    encoder_state = torch.load(encoder_path, map_location=device, weights_only=False)
    model.drug_encoder.load_state_dict(encoder_state['drug_encoder'])
    model.protein_encoder.load_state_dict(encoder_state['protein_encoder'])
    print(f"Loaded pre-trained encoder from {encoder_path}")

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = (sum(p.numel() for p in model.protein_encoder.parameters()) +
                      sum(p.numel() for p in model.drug_encoder.parameters()))
    decoder_params = total_params - encoder_params
    print(f"[{args.decoder}] Total params: {total_params:,} | "
          f"Encoder: {encoder_params:,} | Decoder: {decoder_params:,}")

    best_ckpt_path = os.path.join(log_dir, f'best_pred_{args.decoder}.pt')
    resume_ckpt_path = os.path.join(log_dir, f'resume_pred_{args.decoder}.pt')

    if args.freeze_epochs > 0:
        for p in model.drug_encoder.parameters():
            p.requires_grad = False
        for p in model.protein_encoder.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCELoss()

    best_auc, patience_counter, start_epoch = 0, 0, 0
    unfrozen = args.freeze_epochs == 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('best_auc', 0)
        patience_counter = ckpt.get('patience_counter', 0)
        unfrozen = ckpt.get('unfrozen', args.freeze_epochs == 0)
        print(f"\n*** Resuming from epoch {start_epoch} | best_auc={best_auc:.4f} | "
              f"patience={patience_counter} | unfrozen={unfrozen} ***")
        if 'optimizer_state_dict' in ckpt:
            if unfrozen:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            else:
                for p in model.drug_encoder.parameters():
                    p.requires_grad = False
                for p in model.protein_encoder.parameters():
                    p.requires_grad = False
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            best_auc = best_ckpt.get('best_auc', best_auc)
            print(f"  Restored best_auc={best_auc:.4f} from best checkpoint")

    print(f"\nPhase 2: DTI Fine-tuning ({args.decoder} decoder)")
    print("=" * 60)

    accum_steps = 16

    t_start = time.time()
    for epoch in range(start_epoch, 100):
        if not unfrozen and epoch == args.freeze_epochs:
            print("  Unfreezing encoders...")
            for p in model.drug_encoder.parameters():
                p.requires_grad = True
            for p in model.protein_encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            unfrozen = True

        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        n = len(train_ds)
        indices = np.random.permutation(n)
        optimizer.zero_grad()

        for count, idx in enumerate(tqdm(range(n), desc=f"E{epoch+1}", leave=False)):
            pg, dg, label = train_ds[indices[idx]]
            pg, dg = pg.to(device), dg.to(device)
            out = model(pg, dg)
            loss = criterion(out.view(-1),
                             torch.tensor([label], dtype=torch.float, device=device))
            (loss / accum_steps).backward()
            total_loss += loss.item()
            all_preds.append(out.item())
            all_labels.append(label)
            del pg, dg, out

            if (count + 1) % accum_steps == 0 or (count + 1) == n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
        elapsed = (time.time() - t_start) / 60

        dev_metrics = evaluate(model, dev_ds, device, desc="Dev")
        print(f"[{args.decoder}] E{epoch+1}: Loss={total_loss/n:.4f} "
              f"TrainAUC={train_auc:.4f} DevAUC={dev_metrics['AUC']:.4f} "
              f"F1={dev_metrics['F1']:.4f} ({elapsed:.1f}min)")

        if dev_metrics['AUC'] > best_auc:
            best_auc = dev_metrics['AUC']
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'decoder_type': args.decoder,
                        'encoder_path': encoder_path,
                        'best_auc': best_auc, 'metrics': dev_metrics}, best_ckpt_path)
            print(f"  *** New best (AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'decoder_type': args.decoder,
                    'encoder_path': encoder_path,
                    'best_auc': best_auc, 'patience_counter': patience_counter,
                    'unfrozen': unfrozen, 'metrics': dev_metrics}, resume_ckpt_path)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Final Test Evaluation")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    dev_metrics = evaluate(model, dev_ds, device, desc="Dev Final")
    opt_thresh = dev_metrics['Threshold']

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Test"):
            pg, dg, label = test_ds[idx]
            pg, dg = pg.to(device), dg.to(device)
            out = model(pg, dg)
            test_preds.append(out.item())
            test_labels.append(label)
            del pg, dg, out

    test_preds, test_labels = np.array(test_preds), np.array(test_labels)
    pred_binary = (test_preds >= opt_thresh).astype(int)

    print(f"\n[{args.decoder}] Final Test Results (threshold={opt_thresh:.2f}):")
    print("-" * 40)
    auc_val = roc_auc_score(test_labels, test_preds)
    auprc_val = average_precision_score(test_labels, test_preds)
    f1_val = f1_score(test_labels, pred_binary)
    prec_val = precision_score(test_labels, pred_binary, zero_division=0)
    rec_val = recall_score(test_labels, pred_binary, zero_division=0)
    acc_val = accuracy_score(test_labels, pred_binary)
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_binary, labels=[0, 1]).ravel()
    spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  AUC: {auc_val:.4f}")
    print(f"  AUPRC: {auprc_val:.4f}")
    print(f"  F1: {f1_val:.4f}")
    print(f"  Precision: {prec_val:.4f}")
    print(f"  Recall: {rec_val:.4f}")
    print(f"  Accuracy: {acc_val:.4f}")
    print(f"  Specificity: {spec_val:.4f}")
    print("-" * 40)
    print(f"Best Dev AUC: {best_auc:.4f}")
    print(f"Decoder params: {decoder_params:,}")


if __name__ == '__main__':
    main()
