"""
TransformerCPI model (GCN compound encoder + CNN protein encoder + Transformer decoder)
Adapted from Compare_models/transformerCPI for InterpretableDTIP dataset.
"""
import os
import sys
import json
import math
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_all_data, build_unified_maps, set_seed, SEED,
                          smiles_to_atom_features_adj, protein_seq_to_onehot,
                          AtomFeatureDataset)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'TransformerCPI')


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = self.w_k(key).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = self.w_v(value).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size,
                                               padding=(kernel_size - 1) // 2)
                                    for _ in range(self.n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = conved.permute(0, 2, 1)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ft(trg)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        weighted = trg * norm.unsqueeze(-1)
        pooled = weighted.sum(dim=1)
        out = F.relu(self.fc_1(pooled))
        return self.fc_2(out)


class TransformerCPI(nn.Module):
    def __init__(self, atom_dim=26, protein_dim=22, hid_dim=64,
                 n_layers=3, n_heads=8, pf_dim=256, kernel_size=5,
                 dropout=0.1, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        self.decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
        self.weight_1 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        stdv = 1. / math.sqrt(self.weight_1.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)

    def gcn(self, x, adj):
        support = torch.matmul(x, self.weight_1)
        output = torch.matmul(adj, support)
        support = torch.matmul(output, self.weight_2)
        output = torch.matmul(adj, support)
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    def forward(self, compound, adj, atom_num, protein_oh, protein_len):
        compound = self.gcn(compound, adj)
        enc_src = self.encoder(protein_oh)
        compound_mask, protein_mask = self.make_masks(
            atom_num, protein_len, compound.shape[1], protein_oh.shape[1])
        out = self.decoder(compound, enc_src, compound_mask, protein_mask)
        return out


def find_optimal_threshold(labels, preds):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_binary = (preds >= thresh).astype(int)
        f1 = f1_score(labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
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


@torch.no_grad()
def evaluate(model, loader, device, batch_size=64):
    model.eval()
    all_preds, all_labels = [], []
    samples_buf = []
    for feat, adj, atom_num, protein_oh, protein_len, label in loader:
        for i in range(feat.size(0)):
            samples_buf.append((
                feat[i], adj[i], atom_num[i].item(),
                protein_oh[i], protein_len[i].item(),
                label[i].item()))
            if len(samples_buf) == batch_size:
                preds, labs = _eval_batch(model, samples_buf, device)
                all_preds.extend(preds)
                all_labels.extend(labs)
                samples_buf = []
    if samples_buf:
        preds, labs = _eval_batch(model, samples_buf, device)
        all_preds.extend(preds)
        all_labels.extend(labs)
    opt_thresh = find_optimal_threshold(np.array(all_labels), np.array(all_preds))
    metrics = compute_metrics(all_labels, all_preds, opt_thresh)
    metrics['Threshold'] = opt_thresh
    return metrics


def _eval_batch(model, samples, device):
    N = len(samples)
    max_atom = max(s[2] for s in samples)
    max_prot = max(s[4] for s in samples)
    atom_dim = samples[0][0].shape[1]
    prot_dim = samples[0][3].shape[1]
    compounds = torch.zeros(N, max_atom, atom_dim, device=device)
    adjs = torch.zeros(N, max_atom, max_atom, device=device)
    proteins = torch.zeros(N, max_prot, prot_dim, device=device)
    atom_nums, protein_lens = [], []
    labels = []
    for i, (f, a, an, p, pl, lb) in enumerate(samples):
        compounds[i, :an] = f[:an].to(device)
        adj_i = a[:an, :an].to(device) + torch.eye(an, device=device)
        adjs[i, :an, :an] = adj_i
        proteins[i, :pl] = p[:pl].to(device)
        atom_nums.append(an)
        protein_lens.append(pl)
        labels.append(lb)
    out = model(compounds, adjs, atom_nums, proteins, protein_lens)
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()
    return probs, labels


def train_epoch(model, loader, optimizer, device, batch_accum=8):
    model.train()
    total_loss = 0
    n_samples = 0
    optimizer.zero_grad()
    samples_buf = []
    for feat, adj, atom_num, protein_oh, protein_len, label in loader:
        for i in range(feat.size(0)):
            samples_buf.append((
                feat[i], adj[i], atom_num[i].item(),
                protein_oh[i], protein_len[i].item(),
                label[i].item()))
            if len(samples_buf) == batch_accum:
                loss = _forward_batch(model, samples_buf, device)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * len(samples_buf)
                n_samples += len(samples_buf)
                samples_buf = []
    if samples_buf:
        loss = _forward_batch(model, samples_buf, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(samples_buf)
        n_samples += len(samples_buf)
    return total_loss / max(n_samples, 1)


def _forward_batch(model, samples, device):
    N = len(samples)
    max_atom = max(s[2] for s in samples)
    max_prot = max(s[4] for s in samples)
    atom_dim = samples[0][0].shape[1]
    prot_dim = samples[0][3].shape[1]

    compounds = torch.zeros(N, max_atom, atom_dim, device=device)
    adjs = torch.zeros(N, max_atom, max_atom, device=device)
    proteins = torch.zeros(N, max_prot, prot_dim, device=device)
    labels = torch.zeros(N, dtype=torch.long, device=device)
    atom_nums, protein_lens = [], []
    for i, (f, a, an, p, pl, lb) in enumerate(samples):
        compounds[i, :an] = f[:an].to(device)
        adj_i = a[:an, :an].to(device) + torch.eye(an, device=device)
        adjs[i, :an, :an] = adj_i
        proteins[i, :pl] = p[:pl].to(device)
        labels[i] = lb
        atom_nums.append(an)
        protein_lens.append(pl)
    out = model(compounds, adjs, atom_nums, proteins, protein_lens)
    return F.cross_entropy(out, labels)


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
    print(f"Precomputing features...")

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

    model = TransformerCPI(device=device).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
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
            'epoch': epoch + 1, 'train_loss': train_loss,
            'train': {k: v for k, v in train_metrics.items()},
            'dev': {k: v for k, v in dev_metrics.items()},
            'test': {k: v for k, v in test_metrics.items()},
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train AUC: {train_metrics['AUC']:.4f}, F1: {train_metrics['F1']:.4f}")
        print(f"  Dev   AUC: {dev_metrics['AUC']:.4f}, F1: {dev_metrics['F1']:.4f}")
        print(f"  Test  AUC: {test_metrics['AUC']:.4f}, F1: {test_metrics['F1']:.4f}")

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
