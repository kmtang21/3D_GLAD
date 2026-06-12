#!/usr/bin/env python3
"""
Data preprocessing for MSFF-DTA.
Converts InterpretableDTIP raw data into CSV format for MSFF-DTA training.
"""
import os
import argparse
import numpy as np
import random
import pandas as pd

SEED = 42
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


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


def seq_to_amino_acid_string(int_seq):
    mapping = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
    return ''.join([mapping.get(s, 'X') for s in int_seq])


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


def prepare_msff_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    dev_path = os.path.join(output_dir, 'dev.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    mapping_path = os.path.join(output_dir, 'protein_mapping.csv')
    compound_path = os.path.join(output_dir, 'compound_smiles.csv')

    if (os.path.exists(train_path) and os.path.exists(dev_path) and
            os.path.exists(test_path) and os.path.exists(mapping_path) and
            os.path.exists(compound_path)):
        print("MSFF data files already exist, skipping preparation.")
        return output_dir

    train_data, dev_data, test_data = load_all_data(data_dir)

    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)

    unified_p2i_keys = list(unified_p2i.keys())

    protein_sequences = {}
    for pid, seq_int in zip(unified_p2i_keys, unified_seqs):
        protein_sequences[pid] = seq_to_amino_acid_string(seq_int)

    unique_smiles = list(set(unified_smiles))
    compound_df = pd.DataFrame({'smiles': unique_smiles})
    compound_df.to_csv(compound_path, index=False)
    print(f"compound_smiles.csv: {len(compound_df)} unique SMILES")

    mapping_data = []
    for pid in unified_p2i_keys:
        mapping_data.append({
            'prot_id': pid,
            'sequences': protein_sequences[pid]
        })
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"protein_mapping.csv: {len(mapping_df)} proteins")

    def make_csv(split_data, csv_path):
        smiles_list, seqs_list, c2i, p2i, edges, labels = split_data
        rows = []
        for (chem_id, protein_id), label in zip(edges, labels):
            cidx = c2i[chem_id]
            pidx = p2i[protein_id]
            rows.append({
                'COMPOUND_SMILES': unified_smiles[list(unified_c2i.keys()).index(chem_id)]
                if chem_id in unified_c2i else smiles_list[cidx],
                'PROTEIN_SEQUENCE': protein_sequences[protein_id],
                'PROTEIN_ID': protein_id,
                'CLS_LABEL': label,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"{os.path.basename(csv_path)}: {len(df)} pairs")
        return df

    make_csv(train_data, train_path)
    make_csv(dev_data, dev_path)
    make_csv(test_data, test_path)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='MSFF-DTA data preprocessing')
    parser.add_argument('--data_dir', type=str,
                        default='../../InterpretableDTIP/data',
                        help='Path to InterpretableDTIP data directory')
    parser.add_argument('--output_dir', type=str,
                        default='./pre_data',
                        help='Output directory for processed data')
    args = parser.parse_args()

    set_seed(SEED)
    prepare_msff_data(args.data_dir, args.output_dir)
    print(f"Done. Processed data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
