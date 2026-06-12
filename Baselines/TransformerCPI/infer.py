import os
import sys
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import argparse
import csv
from gensim.models import Word2Vec

from model import (Encoder, Decoder, Predictor, pack,
                    DecoderLayer, SelfAttention, PositionwiseFeedforward)
from data_preprocess import mol_features, seq_to_kmers, get_protein_embedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_w2v_model():
    for d in [os.path.join(BASE_DIR, 'pre_data'), os.path.join(BASE_DIR, 'data')]:
        p = os.path.join(d, 'word2vec.model')
        if os.path.exists(p):
            return Word2Vec.load(p)
    print("Cannot find word2vec.model")
    sys.exit(1)


def build_model(device):
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 9

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                       DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)
    return model


def predict_one(model, atom_feat, adj, protein_emb, device):
    model.eval()
    atom_feat_t = torch.FloatTensor(atom_feat)
    adj_t = torch.FloatTensor(adj)
    protein_t = torch.FloatTensor(protein_emb)
    label_t = torch.LongTensor([0])

    data_pack = pack([atom_feat_t], [adj_t], [protein_t], [label_t], device)
    with torch.no_grad():
        _, predicted_labels, predicted_scores = model(data_pack, train=False)
    return predicted_labels[0], predicted_scores[0]


def main():
    parser = argparse.ArgumentParser(description='Inference with TransformerCPI')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV with SMILES,Target columns')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV with SMILES,Target,Prediction columns')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda:0, cpu, etc.)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    w2v_model = load_w2v_model()
    print('Word2Vec model loaded')

    model = build_model(device)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        for candidate in [
            os.path.join(BASE_DIR, 'logs', 'best_model.pt'),
            os.path.join(BASE_DIR, 'best_model.pt'),
        ]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        else:
            print("Cannot find model checkpoint")
            sys.exit(1)
    print(f'Loading model from {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

    rows = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    skipped = 0
    for i, row in enumerate(rows):
        smiles = row['SMILES']
        target = row['Target']

        try:
            atom_feat, adj = mol_features(smiles)
        except Exception as e:
            print(f"  Skipping row {i}: failed to parse SMILES '{smiles}': {e}")
            skipped += 1
            results.append({'SMILES': smiles, 'Target': target, 'Prediction': 'ERROR'})
            continue

        kmers = seq_to_kmers(target, k=3)
        protein_emb = get_protein_embedding(w2v_model, kmers)

        pred_label, pred_score = predict_one(model, atom_feat, adj, protein_emb, device)
        results.append({'SMILES': smiles, 'Target': target, 'Prediction': float(pred_score)})

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(rows)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SMILES', 'Target', 'Prediction'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. {len(results)} predictions saved to {args.output} ({skipped} skipped)")


if __name__ == '__main__':
    main()
