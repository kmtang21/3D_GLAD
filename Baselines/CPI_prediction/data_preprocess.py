import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser(description='Preprocess CPI prediction data')
    parser.add_argument('--data_dir', type=str, default='../../InterpretableDTIP/data',
                        help='Path to InterpretableDTIP data directory')
    parser.add_argument('--output_dir', type=str, default='./pre_data',
                        help='Output directory for preprocessed pickle files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    train_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, train_data[4], train_data[5])
    dev_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, dev_data[4], dev_data[5])
    test_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, test_data[4], test_data[5])

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    torch.save(train_dataset, os.path.join(args.output_dir, 'train.pkl'))
    torch.save(dev_dataset, os.path.join(args.output_dir, 'dev.pkl'))
    torch.save(test_dataset, os.path.join(args.output_dir, 'test.pkl'))
    print(f"Saved preprocessed datasets to {args.output_dir}")


if __name__ == '__main__':
    main()
