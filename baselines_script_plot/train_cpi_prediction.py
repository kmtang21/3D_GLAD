"""
CPI_prediction model (GNN + Attention-CNN)
Adapted from Compare_models/CPI_prediction for InterpretableDTIP dataset.
"""
import os
import sys
import json
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix, accuracy_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_all_data, build_unified_maps, set_seed, SEED,
                          smiles_to_atom_features_adj, protein_seq_to_onehot,
                          AtomFeatureDataset)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'CPI_prediction')


class CPIPredictionModel(nn.Module):
    def __init__(self, atom_dim=26, protein_dim=22, hidden_dim=64,
                 layer_gnn=3, layer_cnn=3, layer_output=3, window=3):
        super().__init__()
        self.embed_atom = nn.Linear(atom_dim, hidden_dim)
        self.embed_protein = nn.Linear(protein_dim, hidden_dim)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2 * window + 1,
            stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * hidden_dim, 2 * hidden_dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * hidden_dim, 2)
        self.layer_gnn = layer_gnn
        self.layer_cnn = layer_cnn

    def gnn(self, xs, adj, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(adj, hs)
        return torch.unsqueeze(torch.mean(xs, dim=0), 0)

    def attention_cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        return torch.unsqueeze(torch.mean(ys, dim=0), 0)

    def forward(self, atom_feat, adj, atom_num, protein_oh, protein_len):
        atom_emb = torch.relu(self.embed_atom(atom_feat[:atom_num]))
        compound_vec = self.gnn(atom_emb, adj[:atom_num, :atom_num], self.layer_gnn)

        protein_emb = torch.relu(self.embed_protein(protein_oh[:protein_len]))
        protein_vec = self.attention_cnn(compound_vec, protein_emb, self.layer_cnn)

        cat_vec = torch.cat((compound_vec, protein_vec), 1)
        for j in range(len(self.W_out)):
            cat_vec = torch.relu(self.W_out[j](cat_vec))
        interaction = self.W_interaction(cat_vec)
        return interaction


def find_optimal_threshold(labels, preds):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_binary = (preds >= thresh).astype(int)
        f1 = f1_score(labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def compute_metrics(labels, preds, threshold=0.5):
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    if len(set(labels_arr)) <= 1:
        return {'AUC': 0, 'AUPRC': 0, 'F1': 0, 'Precision': 0,
                'Recall': 0, 'Accuracy': 0}
    auc = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)
    pred_binary = (preds_arr >= threshold).astype(int)
    f1 = f1_score(labels_arr, pred_binary, zero_division=0)
    precision = precision_score(labels_arr, pred_binary, zero_division=0)
    recall = recall_score(labels_arr, pred_binary, zero_division=0)
    acc = accuracy_score(labels_arr, pred_binary)
    return {'AUC': auc, 'AUPRC': auprc, 'F1': f1,
            'Precision': precision, 'Recall': recall, 'Accuracy': acc}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for feat, adj, atom_num, protein_oh, protein_len, label in loader:
        feat, adj = feat.to(device), adj.to(device)
        protein_oh = protein_oh.to(device)
        for i in range(feat.size(0)):
            out = model(feat[i], adj[i], atom_num[i],
                        protein_oh[i], protein_len[i])
            prob = F.softmax(out, dim=1)[0, 1].item()
            all_preds.append(prob)
            all_labels.append(label[i].item())
    opt_thresh = find_optimal_threshold(np.array(all_labels), np.array(all_preds))
    metrics = compute_metrics(all_labels, all_preds, opt_thresh)
    metrics['Threshold'] = opt_thresh
    return metrics


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for feat, adj, atom_num, protein_oh, protein_len, label in loader:
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(feat.size(0)):
            out = model(feat[i].to(device), adj[i].to(device),
                        atom_num[i], protein_oh[i].to(device), protein_len[i])
            target = label[i].unsqueeze(0).to(device)
            batch_loss += F.cross_entropy(out, target)
        batch_loss = batch_loss / feat.size(0)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / max(len(loader), 1)


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        _warmup = torch.randn(10, 10, device=device)
        _warmup = torch.mm(_warmup, _warmup)
        torch.cuda.synchronize()
        del _warmup

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'InterpretableDTIP', 'data')
    train_data, dev_data, test_data = load_all_data(data_dir)

    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)

    max_atoms = 100
    max_protein_len = 1000
    print(f"Precomputing features for {len(unified_smiles)} drugs and "
          f"{len(unified_seqs)} proteins...")

    drug_feats = np.zeros((len(unified_smiles), max_atoms, 26), dtype=np.float32)
    drug_adjs = np.zeros((len(unified_smiles), max_atoms, max_atoms), dtype=np.float32)
    drug_atom_nums = np.zeros(len(unified_smiles), dtype=np.int64)
    for i, smi in enumerate(tqdm(unified_smiles, desc="Drug features")):
        feat, adj, n = smiles_to_atom_features_adj(smi, max_atoms)
        if feat is not None:
            drug_feats[i] = feat
            drug_adjs[i] = adj
            drug_atom_nums[i] = n

    protein_onehots = np.zeros((len(unified_seqs), max_protein_len, 22), dtype=np.float32)
    protein_lens = np.zeros(len(unified_seqs), dtype=np.int64)
    for i, seq in enumerate(tqdm(unified_seqs, desc="Protein features")):
        oh, slen = protein_seq_to_onehot(seq, max_protein_len)
        protein_onehots[i] = oh
        protein_lens[i] = slen

    train_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, train_data[4], train_data[5])
    dev_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, dev_data[4], dev_data[5])
    test_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, test_data[4], test_data[5])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    model = CPIPredictionModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 100
    patience = 20
    best_auc = 0
    patience_counter = 0

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, 'metrics.jsonl')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)

        train_metrics = evaluate(model, train_loader, device)
        dev_metrics = evaluate(model, dev_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train': {k: v for k, v in train_metrics.items()},
            'dev': {k: v for k, v in dev_metrics.items()},
            'test': {k: v for k, v in test_metrics.items()},
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train AUC: {train_metrics['AUC']:.4f}, F1: {train_metrics['F1']:.4f}, "
              f"Acc: {train_metrics['Accuracy']:.4f}")
        print(f"  Dev   AUC: {dev_metrics['AUC']:.4f}, F1: {dev_metrics['F1']:.4f}, "
              f"Acc: {dev_metrics['Accuracy']:.4f}")
        print(f"  Test  AUC: {test_metrics['AUC']:.4f}, F1: {test_metrics['F1']:.4f}, "
              f"Acc: {test_metrics['Accuracy']:.4f}")

        if dev_metrics['AUC'] > best_auc:
            best_auc = dev_metrics['AUC']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'metrics': dev_metrics,
            }, os.path.join(LOG_DIR, 'best_model.pt'))
            print(f"  *** Best model saved (Dev AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\nDone. Best Dev AUC:", best_auc)


if __name__ == '__main__':
    main()
