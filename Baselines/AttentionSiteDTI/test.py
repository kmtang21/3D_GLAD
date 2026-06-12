import gc
import os
import sys
import pickle
import random
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import TAGConv, GlobalAttention
from torch_geometric.data import Data, Batch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, recall_score, precision_score,
    accuracy_score
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from layers import MultiHeadAttention

PYG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '3D_GLAD', 'cache', 'attention_data.pkl')
LOCAL_CACHE = os.path.join(os.path.dirname(__file__), 'pre_data', 'attention_data.pkl')
PYG_CACHE = LOCAL_CACHE if os.path.exists(LOCAL_CACHE) else PYG_CACHE
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'logs')


class DTITAG(nn.Module):
    def __init__(self):
        super(DTITAG, self).__init__()
        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, K=2))

        self.ligand_graph_conv = nn.ModuleList()
        self.ligand_graph_conv.append(TAGConv(74, 70, K=2))
        self.ligand_graph_conv.append(TAGConv(70, 65, K=2))
        self.ligand_graph_conv.append(TAGConv(65, 60, K=2))
        self.ligand_graph_conv.append(TAGConv(60, 55, K=2))
        self.ligand_graph_conv.append(TAGConv(55, 31, K=2))

        self.pooling_protein = GlobalAttention(nn.Linear(31, 1))
        self.pooling_ligand = GlobalAttention(nn.Linear(31, 1))
        self.dropout_rate = 0.2
        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=0.0)
        self.fc_in = nn.Linear(8680, 4340)
        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)

    def forward(self, protein_graph, drug_graph, device):
        feature_protein = protein_graph.x
        feature_smile = drug_graph.x
        protein_edge_index = protein_graph.edge_index
        drug_edge_index = drug_graph.edge_index

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(feature_protein, protein_edge_index))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(feature_smile, drug_edge_index))

        protein_reps = self.pooling_protein(feature_protein, protein_graph.batch).view(-1, 31)
        drug_batch = torch.zeros(feature_smile.size(0), dtype=torch.long, device=device)
        ligand_rep = self.pooling_ligand(feature_smile, drug_batch).view(-1, 31)

        sequence = torch.cat((ligand_rep, protein_reps), dim=0).view(1, -1, 31)
        seq_len = sequence.size(1)

        mask = torch.eye(140, dtype=torch.bool).view(1, 140, 140).to(device)
        mask[0, seq_len:140, :] = False
        mask[0, :, seq_len:140] = False
        mask[0, :, seq_len - 1] = True
        mask[0, seq_len - 1, :] = True
        mask[0, seq_len - 1, seq_len - 1] = False

        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - seq_len), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)

        h_0 = torch.zeros(2, 1, 31).to(device)
        c_0 = torch.zeros(2, 1, 31).to(device)

        output, _ = self.bilstm(sequence, (h_0, c_0))
        output = output.permute(1, 0, 2)

        out = self.attention(output, mask=mask)
        out = F.relu(self.fc_in(out.view(-1, out.size(1) * out.size(2))))
        out = torch.sigmoid(self.fc_out(out))
        return out


def evaluate(model, dataset, device, desc="Eval"):
    model.eval()
    y_pred = []
    y_true = []
    losses = []
    criterion = nn.BCELoss()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            pgraph, dgraph, label = dataset[i]
            pgraph = pgraph.to(device)
            dgraph = dgraph.to(device)
            out = model(pgraph, dgraph, device)
            y_pred.append(out.cpu().item())
            y_true.append(label)
            target = torch.tensor([[float(label)]], dtype=torch.float).to(device)
            losses.append(criterion(out, target).item())

    y_pred_arr = np.array(y_pred)
    y_true_arr = np.array(y_true)
    y_pred_c = (y_pred_arr >= 0.5).astype(int)

    return {
        'auroc': roc_auc_score(y_true_arr, y_pred_arr),
        'auprc': average_precision_score(y_true_arr, y_pred_arr),
        'f1': f1_score(y_true_arr, y_pred_c),
        'acc': accuracy_score(y_true_arr, y_pred_c),
        'bal_acc': balanced_accuracy_score(y_true_arr, y_pred_c),
        'precision': precision_score(y_true_arr, y_pred_c, zero_division=0),
        'recall': recall_score(y_true_arr, y_pred_c, zero_division=0),
        'loss': float(np.mean(losses)),
    }


def main():
    parser = argparse.ArgumentParser(description='AttentionSiteDTI Test')
    parser.add_argument('--ckpt', type=str, default=os.path.join(OUTPUT_DIR, 'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g. cuda:0, cpu)')
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DTITAG().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}")

    print(f"Loading test data from {PYG_CACHE}")
    with open(PYG_CACHE, 'rb') as f:
        data = pickle.load(f)
    test_ds = data['test']
    print(f"Test set size: {len(test_ds)}")

    metrics = evaluate(model, test_ds, device, desc="Testing")

    print("\n=== Test Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        'checkpoint': args.ckpt,
        'checkpoint_epoch': ckpt['epoch'],
        'best_auc': ckpt['best_auc'],
        'test_metrics': metrics,
    }
    results_path = os.path.join(OUTPUT_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
