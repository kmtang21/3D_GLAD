import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import RDLogger

from model import DTIModel
from data_preprocess import (process_drug, process_protein_pockets,
                             build_fallback_protein_graph, DRUG_FEAT_DIM)
from torch_geometric.data import Batch

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')


def load_model(ckpt_path, device):
    model = DTIModel(latent_dim=31).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from {ckpt_path}")
    return model


def download_pdb(pdb_id, pdb_dir):
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file):
        return pdb_file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        import urllib.request
        print(f"  Downloading {pdb_id}.pdb ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, pdb_file)
        print("OK")
        return pdb_file
    except Exception as e:
        print(f"FAILED ({e})")
        return None


@torch.no_grad()
def encode_protein_from_pdb(pdb_id, model, pdb_dir, device):
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_file):
        pdb_file = download_pdb(pdb_id, pdb_dir)
        if pdb_file is None:
            return None

    pg = process_protein_pockets(pdb_file)
    if pg is None:
        return None

    pg = pg.to(device)
    protein_feat = model.encode_protein(pg.x, pg.edge_index)
    return {'feat': protein_feat.cpu(), 'batch': pg.batch.cpu()}


@torch.no_grad()
def predict_for_pairs(drug_graphs, protein_cache, model, device, desc="Predicting"):
    results = {}

    for drug_name, dg in tqdm(drug_graphs.items(), desc=desc):
        dg = dg.to(device)
        drug_feat = model.encode_drug(dg.x, dg.edge_index)

        for pdb_id, pcache in protein_cache.items():
            if pcache is None:
                continue
            protein_feat = pcache['feat'].to(device)
            protein_batch = pcache['batch'].to(device)
            score = model.decoder(drug_feat, protein_feat, protein_batch)
            results[(drug_name, pdb_id)] = score.item()

    return results


def main():
    parser = argparse.ArgumentParser(description='3DGLAD Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file. Rows=drug SMILES (index), columns=PDB IDs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (default: input_pred.csv)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to best_model.pt (default: logs/best_model.pt)')
    parser.add_argument('--pdb_dir', type=str, default=None,
                        help='Directory containing PDB files (default: cache/pdbs)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(os.path.dirname(__file__), 'logs', 'best_model.pt')
    if args.pdb_dir is None:
        args.pdb_dir = os.path.join(os.path.dirname(__file__), 'cache', 'pdbs')
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_pred.csv"
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    os.makedirs(args.pdb_dir, exist_ok=True)

    df = pd.read_csv(args.input, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()
    print(f"Input: {df.shape[0]} drugs x {df.shape[1]} proteins")

    drug_smiles = df.index.tolist()
    pdb_ids = df.columns.tolist()

    print(f"Processing {len(drug_smiles)} drug graphs...")
    drug_graphs = {}
    skipped_drugs = []
    for i, smi in enumerate(drug_smiles):
        g = process_drug(smi)
        if g is not None:
            drug_graphs[smi] = g
        else:
            skipped_drugs.append(smi)

    if skipped_drugs:
        print(f"  Warning: {len(skipped_drugs)} drugs failed to parse")

    print(f"Processing {len(pdb_ids)} protein structures...")
    protein_cache = {}
    skipped_proteins = []

    for pdb_id in pdb_ids:
        pcache = encode_protein_from_pdb(pdb_id, model, args.pdb_dir, device)
        if pcache is not None:
            protein_cache[pdb_id] = pcache
        else:
            skipped_proteins.append(pdb_id)

    if skipped_proteins:
        print(f"  Warning: {len(skipped_proteins)} proteins failed (no PDB or no pockets)")
        print(f"  Skipped: {skipped_proteins}")

    results = predict_for_pairs(drug_graphs, protein_cache, model, device)

    pred_df = pd.DataFrame(index=drug_smiles, columns=pdb_ids, dtype=float)
    for (drug_name, pdb_id), score in results.items():
        pred_df.loc[drug_name, pdb_id] = score

    pred_df.to_csv(args.output)
    print(f"\nPredictions saved to {args.output}")


if __name__ == '__main__':
    main()
