import os
import sys
import csv
import argparse
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


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


def protein_str_to_onehot(seq_str, max_len=1000):
    n_aa = len(AMINO_ACIDS)
    seq_str = seq_str[:max_len]
    seq_len = len(seq_str)
    onehot = np.zeros((max_len, n_aa + 1), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, aa in enumerate(seq_str):
        idx = aa_to_idx.get(aa, n_aa)
        onehot[i, idx] = 1.0
    return onehot, seq_len


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


@torch.no_grad()
def predict(model, smiles, protein_seq, device, max_atoms=100, max_protein_len=1000):
    feat, adj, atom_num = smiles_to_atom_features_adj(smiles, max_atoms)
    if feat is None:
        return None

    protein_oh, protein_len = protein_str_to_onehot(protein_seq, max_protein_len)

    feat_t = torch.tensor(feat, dtype=torch.float32).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    protein_oh_t = torch.tensor(protein_oh, dtype=torch.float32).to(device)

    out = model(feat_t, adj_t, atom_num, protein_oh_t, protein_len)
    prob = F.softmax(out, dim=1)[0, 1].item()
    return prob


def main():
    parser = argparse.ArgumentParser(description='CPI Prediction Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (columns: SMILES,Target)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file (columns: SMILES,Target,Prediction)')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CPIPredictionModel().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    rows = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    has_header = len(header) >= 2 and header[0].strip().upper() == 'SMILES'
    if not has_header:
        rows.insert(0, header)

    print(f"Processing {len(rows)} pairs...")

    results = []
    for idx, row in enumerate(rows):
        smiles = row[0].strip()
        target = row[1].strip()

        prob = predict(model, smiles, target, device)
        if prob is None:
            print(f"  Warning: could not parse SMILES '{smiles}' at row {idx + 1}, skipping")
            continue

        results.append([smiles, target, prob])
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(rows)}")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'Target', 'Prediction'])
        for r in results:
            writer.writerow(r)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == '__main__':
    main()
