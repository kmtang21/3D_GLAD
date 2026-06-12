import gc
import os
import sys
import pickle
import random
import time
import json
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

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
GAMMA = 0.90
torch.backends.cudnn.enabled = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_data():
    print(f"Loading cached data from {PYG_CACHE}")
    with open(PYG_CACHE, 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['test'], data['dev']


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


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    train_ds, test_ds, dev_ds = load_data()
    print(f"Dataset sizes: train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}")

    model = DTITAG().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = ExponentialLR(optimizer, gamma=GAMMA)

    metrics_log = []
    best_auc = 0.0
    best_epoch = -1

    indices = list(range(len(train_ds)))

    print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(indices)

        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_start in pbar:
            batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
            batch_outputs = []
            batch_labels = []

            optimizer.zero_grad()

            for idx in batch_indices:
                pgraph, dgraph, label = train_ds[idx]
                pgraph = pgraph.to(DEVICE)
                dgraph = dgraph.to(DEVICE)
                out = model(pgraph, dgraph, DEVICE)
                batch_outputs.append(out)
                batch_labels.append(float(label))

            outputs = torch.cat(batch_outputs, dim=0).to(DEVICE)
            targets = torch.tensor(batch_labels, dtype=torch.float).view(-1, 1).to(DEVICE)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            preds = (outputs.detach() >= 0.5).float()
            epoch_correct += (preds.view(-1).cpu() == torch.tensor(batch_labels)).sum().item()
            epoch_total += len(batch_labels)

            pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}", acc=f"{epoch_correct/epoch_total:.4f}")

        scheduler.step()

        train_loss = np.mean(epoch_losses)
        train_acc = epoch_correct / epoch_total

        dev_metrics = evaluate(model, dev_ds, DEVICE, desc=f"Dev E{epoch+1}")
        test_metrics = evaluate(model, test_ds, DEVICE, desc=f"Test E{epoch+1}")

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'dev_auroc': dev_metrics['auroc'],
            'dev_auprc': dev_metrics['auprc'],
            'dev_f1': dev_metrics['f1'],
            'dev_acc': dev_metrics['acc'],
            'dev_bal_acc': dev_metrics['bal_acc'],
            'dev_precision': dev_metrics['precision'],
            'dev_recall': dev_metrics['recall'],
            'dev_loss': dev_metrics['loss'],
            'test_auroc': test_metrics['auroc'],
            'test_auprc': test_metrics['auprc'],
            'test_f1': test_metrics['f1'],
            'test_acc': test_metrics['acc'],
            'test_bal_acc': test_metrics['bal_acc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_loss': test_metrics['loss'],
        }
        metrics_log.append(epoch_log)

        is_best = ""
        if dev_metrics['auroc'] > best_auc:
            best_auc = dev_metrics['auroc']
            best_epoch = epoch + 1
            ckpt_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'metrics': epoch_log,
            }, ckpt_path)
            is_best = " *BEST*"

        print(
            f"E{epoch+1:3d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"dev_auc={dev_metrics['auroc']:.4f} dev_f1={dev_metrics['f1']:.4f} | "
            f"test_auc={test_metrics['auroc']:.4f} test_f1={test_metrics['f1']:.4f} "
            f"test_auprc={test_metrics['auprc']:.4f}{is_best}"
        )

    log_path = os.path.join(OUTPUT_DIR, 'metrics_log.json')
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nMetrics log saved to {log_path}")
    print(f"Best dev AUC: {best_auc:.4f} at epoch {best_epoch}")

    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    final_metrics = evaluate(model, test_ds, DEVICE, desc="Final Test")
    print("\n=== Final Test Results (Best Dev Model) ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    final_path = os.path.join(OUTPUT_DIR, 'final_test_metrics.json')
    with open(final_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == '__main__':
    train()
