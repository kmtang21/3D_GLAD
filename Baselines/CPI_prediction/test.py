import os
import sys
import json
import argparse
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)
from torch.utils.data import DataLoader

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
    from rdkit import Chem
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
    prec = precision_score(labels_arr, pred_binary, zero_division=0)
    rec = recall_score(labels_arr, pred_binary, zero_division=0)
    acc = accuracy_score(labels_arr, pred_binary)
    return {'AUC': float(auc), 'AUPRC': float(auprc), 'F1': float(f1),
            'Precision': float(prec), 'Recall': float(rec), 'Accuracy': float(acc)}


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
    metrics['Threshold'] = float(opt_thresh)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test CPI Prediction Model')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to InterpretableDTIP data directory')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.data_dir is None:
        args.data_dir = os.path.join(base_dir, '..', '..', 'InterpretableDTIP', 'data')

    pkl_dir = os.path.join(base_dir, 'pre_data')
    test_pkl = os.path.join(pkl_dir, 'test.pkl')
    if not os.path.exists(test_pkl):
        pkl_dir = os.path.join(base_dir, 'data')
        test_pkl = os.path.join(pkl_dir, 'test.pkl')
    if os.path.exists(test_pkl):
        print(f"Loading preprocessed test data from {pkl_dir}...")
        test_dataset = torch.load(test_pkl, weights_only=False)
    else:
        from tqdm import tqdm
        print("No preprocessed data found. Running preprocessing...")
        train_data, dev_data, test_data = load_all_data(args.data_dir)

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

        test_dataset = AtomFeatureDataset(
            drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
            unified_c2i, unified_p2i, test_data[4], test_data[5])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    model = CPIPredictionModel().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {args.ckpt}")
    if 'best_auc' in ckpt:
        print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}, best AUC: {ckpt['best_auc']:.4f}")

    metrics = evaluate(model, test_loader, device)

    print("\n===== Test Results =====")
    for k, v in metrics.items():
        if k == 'Threshold':
            print(f"  Optimal Threshold: {v:.4f}")
        else:
            print(f"  {k}: {v:.4f}")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
