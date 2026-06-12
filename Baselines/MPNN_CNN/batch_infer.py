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

torch.backends.cudnn.enabled = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DP_DIR = os.path.join(BASE_DIR, '..', '..', 'Compare_models', 'DeepPurpose')
sys.path.insert(0, DP_DIR)

from DeepPurpose import utils
from DeepPurpose import DTI as models
from DeepPurpose.utils import (smiles2mpnnfeature, mpnn_feature_collate_func,
                                 trans_protein, protein_2_embed)
from torch.utils.data.dataloader import default_collate


@torch.no_grad()
def precompute_drug_embeddings(model_drug, drug_features, device, batch_size=256):
    all_embs = []
    for start in range(0, len(drug_features), batch_size):
        end = min(start + batch_size, len(drug_features))
        batch = drug_features[start:end]
        mpnn_batch = mpnn_feature_collate_func(batch)
        emb = model_drug(mpnn_batch)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def precompute_protein_embedding(model_protein, prot_embed, device):
    p_tensor = torch.tensor(prot_embed).long().unsqueeze(0)
    return model_protein(p_tensor.float().to(device)).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='../../template_input.csv')
    parser.add_argument('--output', default='infer_MPNN_CNN.csv')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pdb_dir', default='/tmp/pdbs_infer')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = utils.generate_config(
        drug_encoding='MPNN', target_encoding='CNN',
        result_folder='/tmp/dp_batch_infer5', cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100, LR=0.001, batch_size=args.batch_size, hidden_dim_drug=128,
        mpnn_hidden_size=128, mpnn_depth=3,
        cnn_target_filters=[32, 64, 64], cnn_target_kernels=[4, 8, 8],
        num_workers=0, cuda_id=None)
    config['decay'] = 0

    model = models.model_initialize(**config)
    model.device = device
    model.model = model.model.to(device)
    model.binary = True
    ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt_data['model_state_dict'])
    model.model.eval()
    print(f"Loaded {args.ckpt}")

    # Extract sub-models
    model_drug = model.model.model_drug
    model_protein = model.model.model_protein
    predictor = model.model.predictor
    dropout = model.model.dropout

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

    # Step 1: Compute drug features
    print(f"Step 1: Computing {n_drugs} drug features...")
    t0 = time.time()
    drug_features = []
    for smi in tqdm(smiles_list):
        drug_features.append(smiles2mpnnfeature(smi))
    print(f"  Drug features: {time.time()-t0:.1f}s")

    # Step 2: Pre-compute drug embeddings (MPNN forward)
    print(f"Step 2: Computing {n_drugs} drug embeddings (MPNN)...")
    t1 = time.time()
    drug_embs = precompute_drug_embeddings(model_drug, drug_features, device, args.batch_size)
    print(f"  Drug embeddings: {drug_embs.shape}, {time.time()-t1:.1f}s")
    del drug_features

    # Step 3: Pre-compute protein embeddings
    print(f"Step 3: Computing {len(pdb_ids)} protein embeddings (CNN)...")
    prot_embs = {}
    for pid in tqdm(pdb_ids):
        prot_enc = trans_protein(pdb_to_seq[pid])
        prot_embed = protein_2_embed(prot_enc)
        prot_embs[pid] = precompute_protein_embedding(model_protein, prot_embed, device)
    print("  Protein embeddings done")

    # Step 4: Batch classifier inference
    print(f"Step 4: Running classifier on all pairs...")
    result_df = pd.DataFrame(index=smiles_list, columns=pdb_ids, dtype=float)
    total = n_drugs * len(pdb_ids)
    done = 0
    t2 = time.time()

    for pid in tqdm(pdb_ids, desc="Proteins"):
        p_emb = prot_embs[pid]  # (1, hid_dim)
        p_emb_batch = p_emb.expand(n_drugs, -1)  # (n_drugs, hid_dim)

        all_preds = []
        for start in range(0, n_drugs, args.batch_size):
            end = min(start + args.batch_size, n_drugs)
            d_emb = drug_embs[start:end].to(device)
            p_emb_b = p_emb_batch[start:end].to(device)

            v_f = torch.cat((d_emb, p_emb_b), 1)
            for i, layer in enumerate(predictor):
                if i == len(predictor) - 1:
                    v_f = layer(v_f)
                else:
                    v_f = F.relu(dropout(v_f))
                    v_f = layer(v_f)

            probs = torch.sigmoid(v_f).squeeze(-1).detach().cpu().numpy().tolist()
            all_preds.extend(probs if isinstance(probs, list) else [probs])

        for i, smi in enumerate(smiles_list):
            result_df.loc[smi, pid] = all_preds[i] if i < len(all_preds) else np.nan

        done += n_drugs
        elapsed = time.time() - t2
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate / 3600 if rate > 0 else 0
        print(f"  {pid}: {done}/{total}, {rate:.0f}/s, ETA {eta:.1f}h")

    result_df.to_csv(args.output)
    print(f"Saved {args.output}")
    print(f"Step 4 time: {(time.time()-t2)/60:.1f} min")
    print(f"Total time: {(time.time()-t0)/3600:.2f} h")


if __name__ == '__main__':
    main()
