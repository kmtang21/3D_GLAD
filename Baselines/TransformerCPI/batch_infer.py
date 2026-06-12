import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from Bio import SeqIO

torch.backends.cudnn.enabled = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model import Encoder, Decoder, Predictor, pack, DecoderLayer, SelfAttention, PositionwiseFeedforward
from data_preprocess import mol_features, seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec


def load_w2v():
    for d in [os.path.join(BASE_DIR, 'pre_data'), os.path.join(BASE_DIR, 'data')]:
        p = os.path.join(d, 'word2vec.model')
        if os.path.exists(p):
            return Word2Vec.load(p)
    sys.exit("word2vec.model not found")


def build_model(device):
    protein_dim, atom_dim, hid_dim = 100, 34, 64
    n_layers, n_heads, pf_dim, dropout, kernel_size = 3, 8, 256, 0.1, 9
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                       DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='../../template_input.csv')
    parser.add_argument('--output', default='infer_TransformerCPI.csv')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pdb_dir', default='/tmp/pdbs_infer')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    w2v = load_w2v()
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded {args.ckpt}")

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

    print(f"Precomputing {n_drugs} drug features...")
    drug_feats = {}
    for smi in tqdm(smiles_list):
        try:
            feat, adj = mol_features(smi)
            drug_feats[smi] = (torch.FloatTensor(feat), torch.FloatTensor(adj))
        except:
            drug_feats[smi] = None

    print(f"Precomputing {len(pdb_ids)} protein embeddings...")
    prot_embs = {}
    for pid in tqdm(pdb_ids):
        kmers = seq_to_kmers(pdb_to_seq[pid])
        emb = get_protein_embedding(w2v, kmers)
        prot_embs[pid] = torch.FloatTensor(emb)

    result_df = pd.DataFrame(index=smiles_list, columns=pdb_ids, dtype=float)
    total = n_drugs * len(pdb_ids)
    done = 0
    t0 = time.time()

    with torch.no_grad():
        for pid in pdb_ids:
            p_emb = prot_embs[pid]
            all_scores = []

            for start in range(0, n_drugs, args.batch_size):
                end = min(start + args.batch_size, n_drugs)
                batch_smiles = smiles_list[start:end]
                atoms_list, adjs_list, proteins_list, labels_list = [], [], [], []
                valid_indices = []

                for idx, smi in enumerate(batch_smiles):
                    if drug_feats[smi] is not None:
                        feat, adj = drug_feats[smi]
                        atoms_list.append(feat)
                        adjs_list.append(adj)
                        proteins_list.append(p_emb)
                        labels_list.append(torch.LongTensor([0]))
                        valid_indices.append(idx)

                if not atoms_list:
                    all_scores.extend([np.nan] * len(batch_smiles))
                    continue

                data_pack = pack(atoms_list, adjs_list, proteins_list, labels_list, device)
                _, _, scores = model(data_pack, train=False)
                batch_results = [np.nan] * len(batch_smiles)
                for i, vi in enumerate(valid_indices):
                    batch_results[vi] = float(scores[i])
                all_scores.extend(batch_results)

            for i, smi in enumerate(smiles_list):
                result_df.loc[smi, pid] = all_scores[i]
            done += n_drugs
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate / 3600
            print(f"  {pid}: {done}/{total}, {rate:.0f}/s, ETA {eta:.1f}h")

    result_df.to_csv(args.output)
    print(f"Saved {args.output} in {(time.time() - t0) / 3600:.2f}h")


if __name__ == '__main__':
    main()
