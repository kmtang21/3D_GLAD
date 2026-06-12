import os
import sys
import gc
import time
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, accuracy_score)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '3D_GLAD'))
from attention_data import build_datasets, DRUG_FEAT_DIM, PROTEIN_FEAT_DIM
from model import DTIModelV2

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_optimal_threshold(labels, preds):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


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
    parser.add_argument('--conv', type=str, default='GCN',
                        choices=['GCN', 'TAG', 'GATv2', 'GIN', 'Transformer'])
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'InterpretableDTIP', 'data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    print(f"Conv: {args.conv} | Decoder: CrossAttn | NO GAE | Seed: {SEED} | Device: {device}")

    print("Loading datasets...")
    train_ds, test_ds, dev_ds = build_datasets(data_dir)
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}, Test: {len(test_ds)}")

    model = DTIModelV2(conv_type=args.conv, latent_dim=31).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = (sum(p.numel() for p in model.protein_encoder.parameters()) +
                      sum(p.numel() for p in model.drug_encoder.parameters()))
    decoder_params = total_params - encoder_params
    print(f"[{args.conv}] Total: {total_params:,} | Encoder: {encoder_params:,} | Decoder: {decoder_params:,}")

    best_ckpt_path = os.path.join(log_dir, f'best_{args.conv}_crossattn_noGAE.pt')
    resume_ckpt_path = os.path.join(log_dir, f'resume_{args.conv}_crossattn_noGAE.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_auc, patience_counter, start_epoch = 0, 0, 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('best_auc', 0)
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"\n*** Resuming from epoch {start_epoch} | best_auc={best_auc:.4f} | "
              f"patience={patience_counter} ***")
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            best_auc = best_ckpt.get('best_auc', best_auc)
            print(f"  Restored best_auc={best_auc:.4f} from best checkpoint")

    print(f"\nTraining ({args.conv} + CrossAttn, NO GAE pre-training)")
    print("All parameters trainable from epoch 1")
    print("=" * 60)

    accum_steps = 16

    t_start = time.time()
    for epoch in range(start_epoch, 100):
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
        print(f"[{args.conv}] E{epoch+1}: Loss={total_loss/n:.4f} "
              f"TrainAUC={train_auc:.4f} DevAUC={dev_metrics['AUC']:.4f} "
              f"F1={dev_metrics['F1']:.4f} ({elapsed:.1f}min)")

        if dev_metrics['AUC'] > best_auc:
            best_auc = dev_metrics['AUC']
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'conv_type': args.conv,
                        'best_auc': best_auc, 'metrics': dev_metrics}, best_ckpt_path)
            print(f"  *** New best (AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'conv_type': args.conv,
                    'best_auc': best_auc, 'patience_counter': patience_counter,
                    'metrics': dev_metrics}, resume_ckpt_path)

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

    print(f"\n[{args.conv}+CrossAttn] Final Test Results (threshold={opt_thresh:.2f}):")
    print("-" * 40)
    print(f"  AUC: {roc_auc_score(test_labels, test_preds):.4f}")
    print(f"  AUPRC: {average_precision_score(test_labels, test_preds):.4f}")
    print(f"  F1: {f1_score(test_labels, pred_binary):.4f}")
    print(f"  Precision: {precision_score(test_labels, pred_binary, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(test_labels, pred_binary, zero_division=0):.4f}")
    print(f"  Accuracy: {accuracy_score(test_labels, pred_binary):.4f}")
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_binary, labels=[0, 1]).ravel()
    print(f"  Specificity: {tn / (tn + fp) if (tn + fp) > 0 else 0:.4f}")
    print("-" * 40)
    print(f"Best Dev AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
