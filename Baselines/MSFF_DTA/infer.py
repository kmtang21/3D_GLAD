#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from hashlib import md5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSFF_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Compare_models', 'MSFF-DTA')
MSFF_DIR = os.path.abspath(MSFF_DIR)
sys.path.insert(0, MSFF_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data_preprocess import SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='best_model.pt')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pre_data_dir = os.path.join(SCRIPT_DIR, 'pre_data')
    msff_data_dir = os.path.join(SCRIPT_DIR, 'msff_data')
    DATA_DIR = pre_data_dir if os.path.exists(pre_data_dir) else msff_data_dir
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    from preprocessing.protein import ProteinFeatureManager
    from preprocessing.compound import CompoundFeatureManager
    from preprocessing.compound import get_mol_features
    from data import CPIDataset
    from torch.utils.data import DataLoader
    from core import Predictor

    print("Initializing protein feature manager...")
    protein_fm = ProteinFeatureManager(DATA_DIR)
    print("Initializing compound feature manager...")
    compound_fm = CompoundFeatureManager(DATA_DIR)

    input_df = pd.read_csv(args.input)
    required_cols = {'SMILES', 'Target'}
    if not required_cols.issubset(set(input_df.columns)):
        print(f"Input CSV must have columns: {required_cols}")
        return

    protein_id_map = {}
    for seq in input_df['Target'].unique():
        protein_id_map[seq] = "INF_" + md5(seq.encode()).hexdigest()[:12]

    new_proteins = {seq: pid for seq, pid in protein_id_map.items()
                    if pid not in protein_fm.protein_node_feature_dict}
    if new_proteins:
        mapping_path = os.path.join(DATA_DIR, 'protein_mapping.csv')
        existing_map = pd.read_csv(mapping_path)
        existing_ids = set(existing_map['prot_id'].values)
        existing_seqs = set(existing_map['sequences'].values)

        new_rows = []
        for seq, pid in new_proteins.items():
            if seq in existing_seqs:
                for _, row in existing_map.iterrows():
                    if row['sequences'] == seq:
                        protein_id_map[seq] = row['prot_id']
                        break
                continue
            new_rows.append({'prot_id': pid, 'sequences': seq})

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            updated_map = pd.concat([existing_map, new_df], ignore_index=True)
            protein_fm_new = ProteinFeatureManager.__new__(ProteinFeatureManager)
            protein_fm_new.pro_res_table = ProteinFeatureManager(DATA_DIR).pro_res_table
            protein_fm_new.data_path = DATA_DIR
            protein_fm_new.id_to_sequence = {}
            protein_fm_new.fg_dict = {}
            protein_fm_new.protein_fg = {}
            protein_fm_new.protein_fg2 = {}
            protein_fm_new.protein_fg3 = {}
            protein_fm_new.protein_node_feature_dict = {}

            for _, row in updated_map.iterrows():
                seq = row['sequences']
                pid = row['prot_id']
                protein_fm_new.id_to_sequence[pid] = seq
                seq_cut = seq[:1200]
                category_seq = protein_fm_new.protein2category(seq_cut)
                protein_fm_new.get_fg_dict(category_seq)
                protein_fm_new.protein_fg[pid] = protein_fm_new.get_fg(category_seq)
                protein_fm_new.protein_fg2[pid] = protein_fm_new.get_fg(category_seq, padding=2)
                protein_fm_new.protein_fg3[pid] = protein_fm_new.get_fg(category_seq, padding=3)
                protein_fm_new.protein_node_feature_dict[pid] = protein_fm_new.seq_feature(pid)

            for pid in protein_fm.protein_node_feature_dict:
                if pid not in protein_fm_new.protein_node_feature_dict:
                    protein_fm_new.id_to_sequence[pid] = protein_fm.id_to_sequence[pid]
                    protein_fm_new.protein_fg[pid] = protein_fm.protein_fg[pid]
                    protein_fm_new.protein_fg2[pid] = protein_fm.protein_fg2[pid]
                    protein_fm_new.protein_fg3[pid] = protein_fm.protein_fg3[pid]
                    protein_fm_new.protein_node_feature_dict[pid] = protein_fm.protein_node_feature_dict[pid]
                    for fg_key, fg_val in protein_fm.fg_dict.items():
                        if fg_key not in protein_fm_new.fg_dict:
                            protein_fm_new.fg_dict[fg_key] = fg_val

            protein_fm = protein_fm_new

    new_compounds = set()
    for smi in input_df['SMILES'].unique():
        canonical = smi
        try:
            from rdkit import Chem
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        except Exception:
            pass
        if canonical not in compound_fm.compound_graph:
            new_compounds.add(canonical)

    if new_compounds:
        print(f"Processing {len(new_compounds)} new compounds...")
        for smi in new_compounds:
            graph, degree, fp, spatial_pos = get_mol_features(smi)
            compound_fm.compound_graph[smi] = graph
            compound_fm.compound_degree[smi] = degree
            compound_fm.compound_fp[smi] = fp
            compound_fm.compound_spatial_pos[smi] = spatial_pos

    msff_rows = []
    for _, row in input_df.iterrows():
        smi = row['SMILES']
        target = row['Target']
        pid = protein_id_map[target]
        msff_rows.append({
            'COMPOUND_SMILES': smi,
            'PROTEIN_SEQUENCE': target,
            'PROTEIN_ID': pid,
            'CLS_LABEL': 0,
        })
    msff_df = pd.DataFrame(msff_rows)

    tmp_dir = tempfile.mkdtemp(prefix='msff_infer_')
    tmp_csv = os.path.join(tmp_dir, 'infer.csv')

    tmp_mapping_path = os.path.join(tmp_dir, 'protein_mapping.csv')
    mapping_rows = []
    for seq, pid in protein_id_map.items():
        mapping_rows.append({'prot_id': pid, 'sequences': seq})
    pd.DataFrame(mapping_rows).to_csv(tmp_mapping_path, index=False)

    tmp_compound_path = os.path.join(tmp_dir, 'compound_smiles.csv')
    all_smiles = list(set(msff_df['COMPOUND_SMILES'].values))
    pd.DataFrame({'smiles': all_smiles}).to_csv(tmp_compound_path, index=False)

    msff_df.to_csv(tmp_csv, index=False)

    model_args = argparse.Namespace(
        objective='classification',
        batch_size=args.batch_size,
        num_workers=0,
        learning_rate=4e-6,
        max_epochs=100,
        early_stop_round=20,
        root_data_path=tmp_dir,
        decoder_layers=3,
        n_heads=3,
        gnn_layers=3,
        protein_gnn_dim=64,
        compound_gnn_dim=34,
        mol2vec_embedding_dim=300,
        hid_dim=64,
        pf_dim=256,
        dropout=0.2,
        protein_encoder_layers=3,
        protein_encoder_head=4,
        cnn_kernel_size=7,
        protein_dim=64,
        atom_dim=34,
        edge_dim=6,
        protein_embedding_dim=1280,
        compound_embedding_dim=2727,
        seed=SEED,
    )

    dataset = CPIDataset(tmp_csv, protein_fm, compound_fm, model_args)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=dataset.collate_fn,
                        shuffle=False, num_workers=0)

    model = Predictor(model_args).to(device)

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(SCRIPT_DIR, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif hasattr(v, 'to'):
                    batch[k] = v.to(device)
            outputs, _, _ = model(batch)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist()
            all_preds.extend(scores)

    output_df = input_df.copy()
    output_df['Prediction'] = all_preds
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
