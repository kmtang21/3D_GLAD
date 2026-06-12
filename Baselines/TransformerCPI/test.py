import os
import sys
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import pickle
import json
import argparse
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)

from model import (Encoder, Decoder, Predictor, pack,
                    DecoderLayer, SelfAttention, PositionwiseFeedforward)

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_optimal_threshold(labels, preds):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_binary = (preds >= thresh).astype(int)
        f1 = f1_score(labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def compute_metrics(labels, preds):
    preds_arr, labels_arr = np.array(preds), np.array(labels)
    auc_score = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)
    opt_thresh = find_optimal_threshold(labels_arr, preds_arr)
    pred_binary = (preds_arr >= opt_thresh).astype(int)
    return {
        'AUC': auc_score, 'AUPRC': auprc,
        'F1': f1_score(labels_arr, pred_binary, zero_division=0),
        'Precision': precision_score(labels_arr, pred_binary, zero_division=0),
        'Recall': recall_score(labels_arr, pred_binary, zero_division=0),
        'Accuracy': accuracy_score(labels_arr, pred_binary),
        'Threshold': opt_thresh,
    }


def evaluate_model(model, dataset, device):
    model.eval()
    T, S = [], []
    with torch.no_grad():
        for data in dataset:
            adjs, atoms, proteins, labels_list = [], [], [], []
            atom, adj, protein, label = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels_list.append(label)
            data_pack = pack(atoms, adjs, proteins, labels_list, device)
            correct_labels, predicted_labels, predicted_scores = model(data_pack, train=False)
            T.extend(correct_labels)
            S.extend(predicted_scores)
    metrics = compute_metrics(T, S)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test TransformerCPI')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to test.pkl (auto-detected if not set)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint (auto-detected if not set)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    if args.data:
        test_path = args.data
    else:
        for d in [os.path.join(BASE_DIR, 'pre_data'), os.path.join(BASE_DIR, 'data')]:
            p = os.path.join(d, 'test.pkl')
            if os.path.exists(p):
                test_path = p
                break
        else:
            print("Cannot find test.pkl")
            sys.exit(1)
    print(f'Loading test data from {test_path}')

    with open(test_path, 'rb') as f:
        dataset_test = pickle.load(f)
    print(f"Test: {len(dataset_test)} samples")

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
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

    test_metrics = evaluate_model(model, dataset_test, device)

    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
