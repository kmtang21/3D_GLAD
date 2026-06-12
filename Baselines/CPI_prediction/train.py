import os
import sys
import json
import random
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

SEED = 42

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def load_split(data_dir):
    chem_ids = []
    with open(os.path.join(data_dir, 'chem'), 'r') as f:
        for line in f:
            chem_ids.append(line.strip())

    chem_smiles = []
    with open(os.path.join(data_dir, 'chem.repr'), 'r') as f:
        for line in f:
            chem_smiles.append(line.strip())

    protein_ids = []
    with open(os.path.join(data_dir, 'protein'), 'r') as f:
        for line in f:
            protein_ids.append(line.strip())

    protein_seqs = []
    with open(os.path.join(data_dir, 'protein.repr'), 'r') as f:
        for line in f:
            seq = [int(x) for x in line.strip().split()]
            protein_seqs.append(seq)

    chem_id_to_idx = {cid: i for i, cid in enumerate(chem_ids)}
    protein_id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    edges = []
    labels = []
    with open(os.path.join(data_dir, 'edges.pos'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(1)

    with open(os.path.join(data_dir, 'edges.neg'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(0)

    return chem_smiles, protein_seqs, chem_id_to_idx, protein_id_to_idx, edges, labels


def load_all_data(base_dir):
    train_data = load_split(os.path.join(base_dir, 'train'))
    dev_data = load_split(os.path.join(base_dir, 'dev'))
    test_data = load_split(os.path.join(base_dir, 'test'))
    return train_data, dev_data, test_data


def build_unified_maps(train_data, dev_data, test_data):
    train_smiles, train_seqs, train_c2i, train_p2i, _, _ = train_data
    dev_smiles, dev_seqs, dev_c2i, dev_p2i, _, _ = dev_data
    test_smiles, test_seqs, test_c2i, test_p2i, _, _ = test_data

    all_smiles = {}
    all_seqs = {}
    for k, v in train_c2i.items():
        all_smiles[k] = train_smiles[v]
    for k, v in dev_c2i.items():
        all_smiles[k] = dev_smiles[v]
    for k, v in test_c2i.items():
        all_smiles[k] = test_smiles[v]
    for k, v in train_p2i.items():
        all_seqs[k] = train_seqs[v]
    for k, v in dev_p2i.items():
        all_seqs[k] = dev_seqs[v]
    for k, v in test_p2i.items():
        all_seqs[k] = test_seqs[v]

    unified_c2i = {k: i for i, k in enumerate(all_smiles.keys())}
    unified_p2i = {k: i for i, k in enumerate(all_seqs.keys())}
    unified_smiles = [all_smiles[k] for k in unified_c2i.keys()]
    unified_seqs = [all_seqs[k] for k in unified_p2i.keys()]

    return unified_smiles, unified_seqs, unified_c2i, unified_p2i


def seq_to_amino_acid_string(int_seq):
    mapping = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
    return ''.join([mapping.get(s, 'X') for s in int_seq])


def smiles_to_atom_features_adj(smiles, max_atoms=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0

    num_atoms = min(mol.GetNumAtoms(), max_atoms)
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'Si', 'Se', 'I']

    def one_hot(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [1 if s == x else 0 for s in allowable_set]

    def atom_feat(atom):
        symbol = one_hot(atom.GetSymbol(), ATOM_TYPES)
        degree = one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        formal_charge = [atom.GetFormalCharge()]
        num_h = one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        aromatic = [1 if atom.GetIsAromatic() else 0]
        return symbol + degree + formal_charge + num_h + aromatic

    features = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        features.append(atom_feat(atom))

    feat_dim = len(features[0]) if features else 26
    padded_features = np.zeros((max_atoms, feat_dim), dtype=np.float32)
    for i, f in enumerate(features):
        padded_features[i] = f

    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            adj[i][j] = 1.0
            adj[j][i] = 1.0

    return padded_features, adj, num_atoms


def protein_seq_to_onehot(int_seq, max_len=1000):
    n_aa = len(AMINO_ACIDS)
    seq_aa = seq_to_amino_acid_string(int_seq)
    seq_aa = seq_aa[:max_len]
    seq_len = len(seq_aa)
    onehot = np.zeros((max_len, n_aa + 1), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, aa in enumerate(seq_aa):
        idx = aa_to_idx.get(aa, n_aa)
        onehot[i, idx] = 1.0
    return onehot, seq_len


from torch.utils.data import Dataset


class AtomFeatureDataset(Dataset):
    def __init__(self, drug_feats, drug_adjs, drug_atom_nums,
                 protein_onehots, protein_lens,
                 chem_id_to_idx, protein_id_to_idx, edges, labels):
        self.drug_feats = drug_feats
        self.drug_adjs = drug_adjs
        self.drug_atom_nums = drug_atom_nums
        self.protein_onehots = protein_onehots
        self.protein_lens = protein_lens
        self.chem_id_to_idx = chem_id_to_idx
        self.protein_id_to_idx = protein_id_to_idx
        self.edges = edges
        self.labels = labels
        self.indices = list(range(len(edges)))

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        chem_id, protein_id = self.edges[idx]
        label = self.labels[idx]
        cidx = self.chem_id_to_idx[chem_id]
        pidx = self.protein_id_to_idx[protein_id]
        feat = torch.tensor(self.drug_feats[cidx], dtype=torch.float32)
        adj = torch.tensor(self.drug_adjs[cidx], dtype=torch.float32)
        atom_num = self.drug_atom_nums[cidx]
        protein_oh = torch.tensor(self.protein_onehots[pidx], dtype=torch.float32)
        protein_len = self.protein_lens[pidx]
        label = torch.tensor(label, dtype=torch.long)
        return feat, adj, atom_num, protein_oh, protein_len, label


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

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'InterpretableDTIP', 'data')

    pkl_dir = os.path.join(os.path.dirname(__file__), 'pre_data')
    train_pkl = os.path.join(pkl_dir, 'train.pkl')
    if not os.path.exists(train_pkl):
        pkl_dir = os.path.join(os.path.dirname(__file__), 'data')
        train_pkl = os.path.join(pkl_dir, 'train.pkl')
    if os.path.exists(train_pkl):
        print(f"Loading preprocessed data from {pkl_dir}...")
        train_dataset = torch.load(train_pkl)
        dev_dataset = torch.load(os.path.join(pkl_dir, 'dev.pkl'))
        test_dataset = torch.load(os.path.join(pkl_dir, 'test.pkl'))
    else:
        from rdkit import Chem
        print("No preprocessed data found. Running preprocessing...")
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
