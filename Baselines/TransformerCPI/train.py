import os
import sys
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import json
import pickle
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)

from model import (Encoder, Decoder, Predictor, Trainer, Tester, pack,
                    DecoderLayer, SelfAttention, PositionwiseFeedforward)

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
PRE_DATA_DIR = os.path.join(BASE_DIR, 'pre_data')
DATA_DIR = PRE_DATA_DIR if os.path.exists(os.path.join(PRE_DATA_DIR, 'train.pkl')) else os.path.join(BASE_DIR, 'data')


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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    with open(os.path.join(DATA_DIR, 'train.pkl'), 'rb') as f:
        dataset_train = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'dev.pkl'), 'rb') as f:
        dataset_dev = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test.pkl'), 'rb') as f:
        dataset_test = pickle.load(f)

    print(f"Train: {len(dataset_train)}, Dev: {len(dataset_dev)}, Test: {len(dataset_test)}")

    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-3
    weight_decay = 1e-4
    iteration = 100
    kernel_size = 9

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                       DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    trainer = Trainer(model, lr, weight_decay, batch)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, 'metrics.jsonl')

    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
    max_AUC_dev = 0
    patience = 20
    patience_counter = 0

    for epoch in range(1, iteration + 1):
        model.train()
        loss_train = trainer.train(dataset_train, device)
        scheduler.step()

        dev_metrics = evaluate_model(model, dataset_dev, device)
        test_metrics = evaluate_model(model, dataset_test, device)

        log_entry = {
            'epoch': epoch, 'train_loss': loss_train,
            'dev': {k: v for k, v in dev_metrics.items()},
            'test': {k: v for k, v in test_metrics.items()},
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"Epoch {epoch}: Loss={loss_train:.4f} | "
              f"Dev AUC={dev_metrics['AUC']:.4f} F1={dev_metrics['F1']:.4f} | "
              f"Test AUC={test_metrics['AUC']:.4f}")

        if dev_metrics['AUC'] > max_AUC_dev:
            max_AUC_dev = dev_metrics['AUC']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'best_model.pt'))
            print(f"  *** Best model saved (Dev AUC: {max_AUC_dev:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best Dev AUC: {max_AUC_dev:.4f}")


if __name__ == '__main__':
    main()
