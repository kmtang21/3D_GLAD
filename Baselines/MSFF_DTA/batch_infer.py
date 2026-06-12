import os
import sys
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time
import shutil
from tqdm import tqdm
from hashlib import md5
from Bio import SeqIO

torch.backends.cudnn.enabled = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSFF_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'Compare_models', 'MSFF-DTA'))
sys.path.insert(0, MSFF_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data_preprocess import SEED
from preprocessing.protein import ProteinFeatureManager
from preprocessing.compound import CompoundFeatureManager, get_mol_features
from data import CPIDataset
from torch.utils.data import DataLoader
from core import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='../../template_input.csv')
    parser.add_argument('--output', default='infer_MSFF_DTA.csv')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pdb_dir', default='/tmp/pdbs_infer')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    PRE_DATA = os.path.join(SCRIPT_DIR, 'pre_data')
    protein_fm = ProteinFeatureManager(PRE_DATA)
    compound_fm = CompoundFeatureManager(PRE_DATA)
    print("Feature managers loaded")

    df = pd.read_csv(args.template, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()
    smiles_list = df.index.tolist()
    pdb_ids = df.columns.tolist()
    n_drugs = len(smiles_list)
    print(f"{n_drugs} drugs x {len(pdb_ids)} proteins")

    pdb_to_seq = {}
    for pid in pdb_ids:
        for rec in SeqIO.parse(os.path.join(args.pdb_dir, f'{pid}.pdb'), 'pdb-seqres'):
            pdb_to_seq[pid] = str(rec.seq)
            break

    # Pre-register new proteins
    protein_id_map = {}
    for pid, seq in pdb_to_seq.items():
        pid_hash = "INF_" + md5(seq.encode()).hexdigest()[:12]
        protein_id_map[pid] = pid_hash

    new_proteins = {seq: protein_id_map[pid] for pid, seq in pdb_to_seq.items()
                    if protein_id_map[pid] not in protein_fm.protein_node_feature_dict}
    if new_proteins:
        mapping_path = os.path.join(PRE_DATA, 'protein_mapping.csv')
        existing_map = pd.read_csv(mapping_path)
        existing_seqs = set(existing_map['sequences'].values)

        new_fm = ProteinFeatureManager.__new__(ProteinFeatureManager)
        new_fm.pro_res_table = ProteinFeatureManager(PRE_DATA).pro_res_table
        new_fm.data_path = PRE_DATA
        new_fm.id_to_sequence = {}
        new_fm.fg_dict = {}
        new_fm.protein_fg = {}
        new_fm.protein_fg2 = {}
        new_fm.protein_fg3 = {}
        new_fm.protein_node_feature_dict = {}

        updated_map = existing_map.copy()
        for seq, pid in new_proteins.items():
            if seq in existing_seqs:
                for _, row in existing_map.iterrows():
                    if row['sequences'] == seq:
                        protein_id_map[[k for k, v in pdb_to_seq.items() if v == seq][0]] = row['prot_id']
                        break
                continue
            updated_map = pd.concat([updated_map, pd.DataFrame([{'prot_id': pid, 'sequences': seq}])], ignore_index=True)
            new_fm.id_to_sequence[pid] = seq
            seq_cut = seq[:1200]
            category_seq = new_fm.protein2category(seq_cut)
            new_fm.get_fg_dict(category_seq)
            new_fm.protein_fg[pid] = new_fm.get_fg(category_seq)
            new_fm.protein_fg2[pid] = new_fm.get_fg(category_seq, padding=2)
            new_fm.protein_fg3[pid] = new_fm.get_fg(category_seq, padding=3)
            new_fm.protein_node_feature_dict[pid] = new_fm.seq_feature(pid)

        for pid in protein_fm.protein_node_feature_dict:
            if pid not in new_fm.protein_node_feature_dict:
                new_fm.id_to_sequence[pid] = protein_fm.id_to_sequence[pid]
                new_fm.protein_fg[pid] = protein_fm.protein_fg[pid]
                new_fm.protein_fg2[pid] = protein_fm.protein_fg2[pid]
                new_fm.protein_fg3[pid] = protein_fm.protein_fg3[pid]
                new_fm.protein_node_feature_dict[pid] = protein_fm.protein_node_feature_dict[pid]
        protein_fm = new_fm
        print(f"Registered {len(new_proteins)} new proteins")

    # Pre-register new compounds
    from rdkit import Chem
    new_compounds = set()
    for smi in smiles_list:
        try:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        except:
            canonical = smi
        if canonical not in compound_fm.compound_graph:
            new_compounds.add(canonical)
    if new_compounds:
        print(f"Processing {len(new_compounds)} new compounds...")
        for smi in tqdm(new_compounds):
            graph, degree, fp, spatial_pos = get_mol_features(smi)
            compound_fm.compound_graph[smi] = graph
            compound_fm.compound_degree[smi] = degree
            compound_fm.compound_fp[smi] = fp
            compound_fm.compound_spatial_pos[smi] = spatial_pos

    model_args = argparse.Namespace(
        objective='classification', batch_size=args.batch_size, num_workers=0,
        learning_rate=4e-6, max_epochs=100, early_stop_round=20,
        root_data_path=PRE_DATA, decoder_layers=3, n_heads=3, gnn_layers=3,
        protein_gnn_dim=64, compound_gnn_dim=34, mol2vec_embedding_dim=300,
        hid_dim=64, pf_dim=256, dropout=0.2, protein_encoder_layers=3,
        protein_encoder_head=4, cnn_kernel_size=7, protein_dim=64, atom_dim=34,
        edge_dim=6, protein_embedding_dim=1280, compound_embedding_dim=2727, seed=SEED)

    model = Predictor(model_args).to(device)
    ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data['model_state_dict'])
    model.eval()
    print(f"Loaded {args.ckpt}")

    result_df = pd.DataFrame(index=smiles_list, columns=pdb_ids, dtype=float)
    total = n_drugs * len(pdb_ids)
    done = 0
    t0 = time.time()

    for pid in tqdm(pdb_ids, desc="Proteins"):
        seq = pdb_to_seq[pid]
        p_id = protein_id_map[pid]

        tmp_dir = tempfile.mkdtemp(prefix='msff_batch_')
        tmp_csv = os.path.join(tmp_dir, 'infer.csv')

        rows = []
        for smi in smiles_list:
            rows.append({
                'COMPOUND_SMILES': smi, 'PROTEIN_SEQUENCE': seq,
                'PROTEIN_ID': p_id, 'CLS_LABEL': 0})
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)

        dataset = CPIDataset(tmp_csv, protein_fm, compound_fm, model_args)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                           collate_fn=dataset.collate_fn, shuffle=False, num_workers=0)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    elif hasattr(v, 'to'):
                        batch[k] = v.to(device)
                outputs, _, _ = model(batch)
                scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist()
                all_preds.extend(scores)

        for i, smi in enumerate(smiles_list):
            result_df.loc[smi, pid] = all_preds[i] if i < len(all_preds) else np.nan

        shutil.rmtree(tmp_dir, ignore_errors=True)
        done += n_drugs
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate / 3600
        print(f"  {pid}: {done}/{total}, {rate:.0f}/s, ETA {eta:.1f}h")

    result_df.to_csv(args.output)
    print(f"Saved {args.output} in {(time.time() - t0) / 3600:.2f}h")


if __name__ == '__main__':
    main()
