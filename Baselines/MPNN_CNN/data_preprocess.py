"""
Preprocess InterpretableDTIP data into DeepPurpose CSV format.
Output: train.csv, dev.csv, test.csv with columns: SMILES, Target, Label
"""
import os
import csv
import argparse
import numpy as np

SEED = 42
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


def build_csv_data(split_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i):
    _, _, _, _, edges, labels = split_data
    rows = []
    for (chem_id, protein_id), label in zip(edges, labels):
        cidx = unified_c2i.get(chem_id)
        pidx = unified_p2i.get(protein_id)
        if cidx is not None and pidx is not None:
            smiles = unified_smiles[cidx]
            seq = seq_to_amino_acid_string(unified_seqs[pidx])
            rows.append((smiles, seq, label))
    return rows


def save_csv(rows, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'Target', 'Label'])
        for smiles, target, label in rows:
            writer.writerow([smiles, target, label])


def main():
    parser = argparse.ArgumentParser(description='Convert InterpretableDTIP data to DeepPurpose CSV format')
    parser.add_argument('--data_dir', type=str, default='../../InterpretableDTIP/data',
                        help='Path to InterpretableDTIP data directory')
    parser.add_argument('--output_dir', type=str, default='./pre_data',
                        help='Output directory for CSV files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.output_dir, 'train.csv')
    dev_csv = os.path.join(args.output_dir, 'dev.csv')
    test_csv = os.path.join(args.output_dir, 'test.csv')

    if os.path.exists(train_csv) and os.path.exists(dev_csv) and os.path.exists(test_csv):
        print(f"Pre-processed data already exists in {args.output_dir}, skipping.")
        return

    print(f"Loading data from {args.data_dir}...")
    train_data, dev_data, test_data = load_all_data(args.data_dir)

    print("Building unified maps...")
    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)

    print("Converting splits...")
    train_rows = build_csv_data(train_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i)
    dev_rows = build_csv_data(dev_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i)
    test_rows = build_csv_data(test_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i)

    save_csv(train_rows, os.path.join(args.output_dir, 'train.csv'))
    save_csv(dev_rows, os.path.join(args.output_dir, 'dev.csv'))
    save_csv(test_rows, os.path.join(args.output_dir, 'test.csv'))

    print(f"Train: {len(train_rows)} pairs")
    print(f"Dev:   {len(dev_rows)} pairs")
    print(f"Test:  {len(test_rows)} pairs")
    print(f"Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
