import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from Bio import SeqIO
from rdkit import Chem

torch.backends.cudnn.enabled = False

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'Si', 'Se', 'I']


def one_hot(x, s):
    if x not in s: x = s[-1]
    return [1 if c == x else 0 for c in s]


def smiles_to_feat(smiles, max_atoms=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0
    num_atoms = min(mol.GetNumAtoms(), max_atoms)
    feats = np.zeros((max_atoms, 26), dtype=np.float32)
    for i in range(num_atoms):
        a = mol.GetAtomWithIdx(i)
        feats[i] = (one_hot(a.GetSymbol(), ATOM_TYPES) + one_hot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
                     + [a.GetFormalCharge()] + one_hot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
                     + [1 if a.GetIsAromatic() else 0])
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            adj[i][j] = adj[j][i] = 1.0
    return feats, adj, num_atoms


def seq_to_onehot(seq, max_len=1000):
    n_aa = len(AMINO_ACIDS)
    seq = seq[:max_len]; slen = len(seq)
    oh = np.zeros((max_len, n_aa + 1), dtype=np.float32)
    aa_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, aa in enumerate(seq):
        oh[i, aa_map.get(aa, n_aa)] = 1.0
    return oh, slen


class CPIPredictionModel(nn.Module):
    def __init__(self, atom_dim=26, protein_dim=22, hidden_dim=64,
                 layer_gnn=3, layer_cnn=3, layer_output=3, window=3):
        super().__init__()
        self.embed_atom = nn.Linear(atom_dim, hidden_dim)
        self.embed_protein = nn.Linear(protein_dim, hidden_dim)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=2 * window + 1,
                                               stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * hidden_dim, 2 * hidden_dim) for _ in range(layer_output)])
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

    def compute_compound_vec(self, atom_feat, adj, atom_num):
        atom_emb = torch.relu(self.embed_atom(atom_feat[:atom_num]))
        return self.gnn(atom_emb, adj[:atom_num, :atom_num], self.layer_gnn)

    def compute_protein_emb(self, protein_oh, protein_len):
        return torch.relu(self.embed_protein(protein_oh[:protein_len]))

    def predict_from_vecs(self, compound_vec, protein_emb):
        protein_vec = self.attention_cnn(compound_vec, protein_emb, self.layer_cnn)
        cat_vec = torch.cat((compound_vec, protein_vec), 1)
        for j in range(len(self.W_out)):
            cat_vec = torch.relu(self.W_out[j](cat_vec))
        return self.W_interaction(cat_vec)

    def forward(self, atom_feat, adj, atom_num, protein_oh, protein_len):
        compound_vec = self.compute_compound_vec(atom_feat, adj, atom_num)
        protein_emb = self.compute_protein_emb(protein_oh, protein_len)
        return self.predict_from_vecs(compound_vec, protein_emb)

    def batch_predict(self, compound_vecs, protein_emb, batch_size=512):
        n = compound_vecs.shape[0]
        all_probs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_cv = compound_vecs[start:end]
            protein_vec = self.attention_cnn_single(batch_cv, protein_emb, self.layer_cnn)
            cat_vec = torch.cat((batch_cv, protein_vec), 1)
            for j in range(len(self.W_out)):
                cat_vec = torch.relu(self.W_out[j](cat_vec))
            out = self.W_interaction(cat_vec)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.append(probs.cpu())
        return torch.cat(all_probs)

    def attention_cnn_single(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        return torch.mean(ys, dim=0, keepdim=False).unsqueeze(0) if ys.shape[0] == 1 else torch.mean(ys * weights.t(), dim=1, keepdim=True).squeeze(1)


def get_pdb_sequences(pdb_ids, pdb_dir):
    pdb_to_seq = {}
    for pid in pdb_ids:
        pdb_file = os.path.join(pdb_dir, f'{pid}.pdb')
        for record in SeqIO.parse(pdb_file, 'pdb-seqres'):
            pdb_to_seq[pid] = str(record.seq)
            break
    return pdb_to_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='../../template_input.csv')
    parser.add_argument('--output', default='infer_CPI_prediction.csv')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pdb_dir', default='/tmp/pdbs_infer')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CPIPredictionModel().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded {args.ckpt}")

    df = pd.read_csv(args.template, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()
    smiles_list = df.index.tolist()
    pdb_ids = df.columns.tolist()
    n_drugs = len(smiles_list)
    print(f"{n_drugs} drugs x {len(pdb_ids)} proteins = {n_drugs * len(pdb_ids)} pairs")

    pdb_to_seq = get_pdb_sequences(pdb_ids, args.pdb_dir)

    print(f"Precomputing {n_drugs} compound vectors (GNN)...")
    compound_vecs = torch.zeros(n_drugs, 64, device=device)
    valid_mask = torch.ones(n_drugs, dtype=torch.bool)
    t0 = time.time()
    with torch.no_grad():
        for i, smi in enumerate(tqdm(smiles_list)):
            f, a, n = smiles_to_feat(smi)
            if f is not None:
                cv = model.compute_compound_vec(
                    torch.tensor(f).to(device),
                    torch.tensor(a).to(device), n)
                compound_vecs[i] = cv.squeeze(0)
            else:
                valid_mask[i] = False
    print(f"Compound vectors: {time.time() - t0:.1f}s")

    print(f"Precomputing {len(pdb_ids)} protein embeddings...")
    prot_embs = {}
    with torch.no_grad():
        for pid in tqdm(pdb_ids):
            oh, slen = seq_to_onehot(pdb_to_seq[pid])
            pe = model.compute_protein_emb(torch.tensor(oh).to(device), slen)
            prot_embs[pid] = pe

    result_df = pd.DataFrame(index=smiles_list, columns=pdb_ids, dtype=float)
    t0 = time.time()
    total_pairs = n_drugs * len(pdb_ids)
    done = 0

    with torch.no_grad():
        for pid in pdb_ids:
            pe = prot_embs[pid]
            all_probs = []
            for start in range(0, n_drugs, args.batch_size):
                end = min(start + args.batch_size, n_drugs)
                batch_cv = compound_vecs[start:end]
                batch_valid = valid_mask[start:end]

                pe_expanded = pe.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, pe.shape[-1])
                pe_4d = pe.unsqueeze(0).unsqueeze(0)
                for i in range(len(model.W_cnn)):
                    pe_4d = torch.relu(model.W_cnn[i](pe_4d))
                pe_cnn = pe_4d.squeeze(0).squeeze(0)

                h = torch.relu(model.W_attention(batch_cv))
                hs = torch.relu(model.W_attention(pe_cnn))
                weights = torch.tanh(F.linear(h, hs))
                ys = weights.unsqueeze(2) * hs.unsqueeze(0)
                protein_vecs = torch.mean(ys, dim=1)

                cat_vec = torch.cat((batch_cv, protein_vecs), 1)
                for j in range(len(model.W_out)):
                    cat_vec = torch.relu(model.W_out[j](cat_vec))
                out = model.W_interaction(cat_vec)
                probs = F.softmax(out, dim=1)[:, 1]
                probs = probs.masked_fill(~batch_valid.to(device), float('nan'))
                all_probs.append(probs.cpu())

            all_probs = torch.cat(all_probs).numpy()
            for i, smi in enumerate(smiles_list):
                result_df.loc[smi, pid] = all_probs[i]
            done += n_drugs
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total_pairs - done) / rate / 3600
            print(f"  {pid}: {done}/{total_pairs}, {rate:.0f}/s, ETA {eta:.1f}h")

    result_df.to_csv(args.output)
    print(f"Saved {args.output} in {(time.time() - t0) / 3600:.2f}h")


if __name__ == '__main__':
    main()
