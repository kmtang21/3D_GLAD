import os
import sys
import argparse
import pickle
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from gensim.models import Word2Vec
from tqdm import tqdm


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"

SEED = 42
num_atom_feat = 34


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
              [atom.GetIsAromatic()]

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                   [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=np.float32)


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i + k] for i in range(N - k + 1)]


def get_protein_embedding(model, protein):
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        if word in model.wv:
            vec[i, ] = model.wv[word]
        i += 1
    return vec


def seq_to_amino_acid_string(int_seq):
    mapping = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
    return ''.join([mapping.get(s, 'X') for s in int_seq])


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


def train_word2vec(unified_seqs):
    print("Training word2vec on protein sequences...")
    corpus = []
    for seq_int in unified_seqs:
        seq_aa = seq_to_amino_acid_string(seq_int)
        kmers = seq_to_kmers(seq_aa, 3)
        corpus.append(kmers)
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=6, seed=SEED)
    model.build_vocab(corpus)
    model.train(corpus, epochs=30, total_examples=model.corpus_count)
    return model


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for TransformerCPI')
    parser.add_argument('--data_dir', type=str, default='../../InterpretableDTIP/data',
                        help='Path to InterpretableDTIP data directory')
    parser.add_argument('--output_dir', type=str, default='./pre_data',
                        help='Output directory for preprocessed pkl files')
    parser.add_argument('--w2v_model', type=str, default=None,
                        help='Path to existing word2vec model (will train if not provided)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data, dev_data, test_data = load_all_data(args.data_dir)
    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)

    if args.w2v_model and os.path.exists(args.w2v_model):
        print(f"Loading word2vec model from {args.w2v_model}...")
        w2v_model = Word2Vec.load(args.w2v_model)
    else:
        w2v_path = os.path.join(args.output_dir, 'word2vec.model')
        if os.path.exists(w2v_path):
            print(f"Loading cached word2vec model from {w2v_path}...")
            w2v_model = Word2Vec.load(w2v_path)
        else:
            w2v_model = train_word2vec(unified_seqs)
            w2v_model.save(w2v_path)
            print(f"Word2Vec model saved to {w2v_path}, vocab size: {len(w2v_model.wv)}")

    def process_split(split_data, split_name):
        edges = split_data[4]
        labels = split_data[5]
        compounds, adjacencies, proteins, interactions = [], [], [], []

        print(f"Processing {split_name} ({len(edges)} samples)...")
        skipped = 0
        for idx, ((chem_id, prot_id), label) in enumerate(tqdm(zip(edges, labels), total=len(edges))):
            cidx = unified_c2i[chem_id]
            pidx = unified_p2i[prot_id]
            smiles = unified_smiles[cidx]
            seq_int = unified_seqs[pidx]
            seq_aa = seq_to_amino_acid_string(seq_int)

            try:
                atom_feature, adj = mol_features(smiles)
            except Exception as e:
                skipped += 1
                continue

            protein_embedding = get_protein_embedding(w2v_model, seq_to_kmers(seq_aa))

            atom_feature = torch.FloatTensor(atom_feature)
            adj = torch.FloatTensor(adj)
            protein = torch.FloatTensor(protein_embedding)
            lbl = torch.LongTensor([label])

            compounds.append(atom_feature)
            adjacencies.append(adj)
            proteins.append(protein)
            interactions.append(lbl)

        dataset = list(zip(compounds, adjacencies, proteins, interactions))
        out_path = os.path.join(args.output_dir, f'{split_name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  Saved {len(dataset)} samples to {out_path} (skipped {skipped})")
        return dataset

    train_ds = process_split(train_data, 'train')
    dev_ds = process_split(dev_data, 'dev')
    test_ds = process_split(test_data, 'test')

    print(f"Done. Train={len(train_ds)}, Dev={len(dev_ds)}, Test={len(test_ds)}")


if __name__ == '__main__':
    main()
