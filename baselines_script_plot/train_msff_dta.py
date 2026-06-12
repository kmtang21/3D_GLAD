"""
MSFF-DTA model using original code from Compare_models/MSFF-DTA.
Adapted for InterpretableDTIP dataset.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)
from tqdm import tqdm

MSFF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'Compare_models', 'MSFF-DTA')
sys.path.insert(0, MSFF_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (load_all_data, build_unified_maps, set_seed, SEED,
                          seq_to_amino_acid_string)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'MSFF_DTA')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'msff_data')


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def prepare_msff_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, 'train.csv')
    dev_path = os.path.join(DATA_DIR, 'dev.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    mapping_path = os.path.join(DATA_DIR, 'protein_mapping.csv')
    compound_path = os.path.join(DATA_DIR, 'compound_smiles.csv')

    if (os.path.exists(train_path) and os.path.exists(dev_path) and
            os.path.exists(test_path) and os.path.exists(mapping_path) and
            os.path.exists(compound_path)):
        print("MSFF data files already exist, skipping preparation.")
        return DATA_DIR

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'InterpretableDTIP', 'data')
    train_data, dev_data, test_data = load_all_data(data_dir)

    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)

    unified_p2i_keys = list(unified_p2i.keys())

    protein_sequences = {}
    for pid, seq_int in zip(unified_p2i_keys, unified_seqs):
        protein_sequences[pid] = seq_to_amino_acid_string(seq_int)

    unique_smiles = list(set(unified_smiles))
    compound_df = pd.DataFrame({'smiles': unique_smiles})
    compound_df.to_csv(compound_path, index=False)
    print(f"compound_smiles.csv: {len(compound_df)} unique SMILES")

    mapping_data = []
    for pid in unified_p2i_keys:
        mapping_data.append({
            'prot_id': pid,
            'sequences': protein_sequences[pid]
        })
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"protein_mapping.csv: {len(mapping_df)} proteins")

    def make_csv(split_data, csv_path):
        smiles_list, seqs_list, c2i, p2i, edges, labels = split_data
        rows = []
        for (chem_id, protein_id), label in zip(edges, labels):
            cidx = c2i[chem_id]
            pidx = p2i[protein_id]
            rows.append({
                'COMPOUND_SMILES': unified_smiles[list(unified_c2i.keys()).index(chem_id)]
                if chem_id in unified_c2i else smiles_list[cidx],
                'PROTEIN_SEQUENCE': protein_sequences[protein_id],
                'PROTEIN_ID': protein_id,
                'CLS_LABEL': label,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"{os.path.basename(csv_path)}: {len(df)} pairs")
        return df

    make_csv(train_data, train_path)
    make_csv(dev_data, dev_path)
    make_csv(test_data, test_path)

    return DATA_DIR


def find_optimal_threshold(labels, preds):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_binary = (preds >= thresh).astype(int)
        f1v = f1_score(labels, pred_binary, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_thresh = thresh
    return best_thresh


def compute_metrics(labels, preds, threshold=0.5):
    preds_arr, labels_arr = np.array(preds), np.array(labels)
    if len(set(labels_arr)) <= 1:
        return {'AUC': 0, 'AUPRC': 0, 'F1': 0, 'Precision': 0,
                'Recall': 0, 'Accuracy': 0}
    auc = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)
    pred_binary = (preds_arr >= threshold).astype(int)
    return {
        'AUC': auc, 'AUPRC': auprc,
        'F1': f1_score(labels_arr, pred_binary, zero_division=0),
        'Precision': precision_score(labels_arr, pred_binary, zero_division=0),
        'Recall': recall_score(labels_arr, pred_binary, zero_division=0),
        'Accuracy': accuracy_score(labels_arr, pred_binary),
    }


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = prepare_msff_data()

    from preprocessing.protein import ProteinFeatureManager
    from preprocessing.compound import CompoundFeatureManager
    from data import CPIDataset
    from torch.utils.data import DataLoader

    print("Initializing protein feature manager...")
    protein_fm = ProteinFeatureManager(data_path)
    print("Initializing compound feature manager...")
    compound_fm = CompoundFeatureManager(data_path)

    args = argparse.Namespace(
        objective='classification',
        batch_size=64,
        num_workers=0,
        learning_rate=4e-6,
        max_epochs=100,
        early_stop_round=20,
        root_data_path=data_path,
        decoder_layers=3,
        n_heads=3,
        gnn_layers=3,
        protein_gnn_dim=64,
        compound_gnn_dim=34,
        mol2vec_embedding_dim=300,
        hid_dim=64,
        pf_dim=256,
        dropout=0.2,
        protein_encoder_layers=3,
        protein_encoder_head=4,
        cnn_kernel_size=7,
        protein_dim=64,
        atom_dim=34,
        edge_dim=6,
        protein_embedding_dim=1280,
        compound_embedding_dim=2727,
        seed=SEED,
    )

    train_csv = os.path.join(data_path, 'train.csv')
    dev_csv = os.path.join(data_path, 'dev.csv')
    test_csv = os.path.join(data_path, 'test.csv')

    train_dataset = CPIDataset(train_csv, protein_fm, compound_fm, args)
    dev_dataset = CPIDataset(dev_csv, protein_fm, compound_fm, args)
    test_dataset = CPIDataset(test_csv, protein_fm, compound_fm, args)

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True, num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                            collate_fn=dev_dataset.collate_fn,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False, num_workers=0)

    from core import Predictor
    model = Predictor(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.max_epochs
    patience = args.early_stop_round
    best_auc = 0
    patience_counter = 0

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, 'metrics.jsonl')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            labels = batch["LABEL"].long().to(device)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif hasattr(v, 'to'):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            outputs, r_att, _ = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        all_split_metrics = {}
        for split_name, loader in [('train', train_loader), ('dev', dev_loader), ('test', test_loader)]:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    labels_t = batch["LABEL"].long()
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                        elif hasattr(v, 'to'):
                            batch[k] = v.to(device)
                    outputs, _, _ = model(batch)
                    scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist()
                    all_preds.extend(scores)
                    all_labels.extend(labels_t.numpy().tolist())

            opt_thresh = find_optimal_threshold(np.array(all_labels), np.array(all_preds))
            metrics = compute_metrics(all_labels, all_preds, opt_thresh)
            metrics['Threshold'] = opt_thresh
            all_split_metrics[split_name] = metrics

        log_entry = {
            'epoch': epoch + 1, 'train_loss': avg_loss,
            'train': {k: v for k, v in all_split_metrics['train'].items()},
            'dev': {k: v for k, v in all_split_metrics['dev'].items()},
            'test': {k: v for k, v in all_split_metrics['test'].items()},
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        dev_metrics = all_split_metrics['dev']
        test_metrics = all_split_metrics['test']
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | "
              f"Dev AUC={dev_metrics['AUC']:.4f} F1={dev_metrics['F1']:.4f} "
              f"Acc={dev_metrics['Accuracy']:.4f} | "
              f"Test AUC={test_metrics['AUC']:.4f}")

        if dev_metrics['AUC'] > best_auc:
            best_auc = dev_metrics['AUC']
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_auc': best_auc, 'metrics': dev_metrics},
                       os.path.join(LOG_DIR, 'best_model.pt'))
            print(f"  *** Best model saved (Dev AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nDone. Best Dev AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
