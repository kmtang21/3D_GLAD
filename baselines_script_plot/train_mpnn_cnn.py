"""
MPNN_CNN model using original DeepPurpose code from Compare_models/DeepPurpose.
Adapted for InterpretableDTIP dataset.
"""
import os
import sys
import json
import copy
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

DEEPPURPOSE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'Compare_models', 'DeepPurpose')
sys.path.insert(0, DEEPPURPOSE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (load_all_data, build_unified_maps, set_seed, SEED,
                          seq_to_amino_acid_string)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'MPNN_CNN')


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def find_optimal_threshold(labels, preds):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_binary = (preds >= thresh).astype(int)
        f1v = f1_score(labels, pred_binary, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_thresh = thresh
    return best_thresh


def compute_metrics(labels, preds, threshold=0.5):
    preds_arr, labels_arr = np.array(preds), np.array(labels)
    if len(set(labels_arr)) <= 1:
        return {'AUC': 0, 'AUPRC': 0, 'F1': 0, 'Precision': 0,
                'Recall': 0, 'Accuracy': 0}
    auc = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)
    pred_binary = (preds_arr >= threshold).astype(int)
    return {
        'AUC': auc, 'AUPRC': auprc,
        'F1': f1_score(labels_arr, pred_binary, zero_division=0),
        'Precision': precision_score(labels_arr, pred_binary, zero_division=0),
        'Recall': recall_score(labels_arr, pred_binary, zero_division=0),
        'Accuracy': accuracy_score(labels_arr, pred_binary),
    }


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from DeepPurpose import utils
    from DeepPurpose import DTI as models

    drug_encoding = 'MPNN'
    target_encoding = 'CNN'

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'InterpretableDTIP', 'data')
    train_data_raw, dev_data_raw, test_data_raw = load_all_data(data_dir)

    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data_raw, dev_data_raw, test_data_raw)

    def get_smiles(chem_id):
        cidx = unified_c2i.get(chem_id)
        if cidx is not None:
            return unified_smiles[cidx]
        return None

    def get_protein_seq(protein_id):
        pidx = unified_p2i.get(protein_id)
        if pidx is not None:
            return seq_to_amino_acid_string(unified_seqs[pidx])
        return None

    def build_df(split_data):
        smiles_list, seqs_list, c2i, p2i, edges, labels = split_data
        drug_smiles = []
        protein_seqs = []
        y = []
        for (chem_id, protein_id), label in zip(edges, labels):
            smi = get_smiles(chem_id)
            seq = get_protein_seq(protein_id)
            if smi is not None and seq is not None:
                drug_smiles.append(smi)
                protein_seqs.append(seq)
                y.append(label)
        return drug_smiles, protein_seqs, np.array(y)

    print("Converting data to DeepPurpose format...")
    train_drugs, train_prots, train_y = build_df(train_data_raw)
    dev_drugs, dev_prots, dev_y = build_df(dev_data_raw)
    test_drugs, test_prots, test_y = build_df(test_data_raw)

    print(f"Train: {len(train_y)}, Dev: {len(dev_y)}, Test: {len(test_y)}")

    train_df = utils.data_process(train_drugs, train_prots, train_y,
                                  drug_encoding, target_encoding,
                                  split_method='no_split', random_seed=SEED)
    dev_df = utils.data_process(dev_drugs, dev_prots, dev_y,
                                drug_encoding, target_encoding,
                                split_method='no_split', random_seed=SEED)
    test_df = utils.data_process(test_drugs, test_prots, test_y,
                                 drug_encoding, target_encoding,
                                 split_method='no_split', random_seed=SEED)

    result_folder = os.path.join(LOG_DIR, 'dp_results')
    os.makedirs(result_folder, exist_ok=True)

    config = utils.generate_config(
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        result_folder=result_folder,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100,
        LR=0.001,
        batch_size=256,
        hidden_dim_drug=128,
        mpnn_hidden_size=128,
        mpnn_depth=3,
        cnn_target_filters=[32, 64, 64],
        cnn_target_kernels=[4, 8, 8],
        num_workers=0,
        cuda_id=None,
    )
    config['decay'] = 0

    model = models.model_initialize(**config)
    model.device = device
    model.model = model.model.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    num_epochs = config['train_epoch']
    patience = 20
    best_auc = 0
    patience_counter = 0

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, 'metrics.jsonl')

    from DeepPurpose.utils import data_process_loader, mpnn_collate_func
    from torch.utils.data import SequentialSampler

    params = {'batch_size': config['batch_size'], 'shuffle': True,
              'num_workers': 0, 'drop_last': False,
              'collate_fn': mpnn_collate_func}
    params_test = {'batch_size': config['batch_size'], 'shuffle': False,
                   'num_workers': 0, 'drop_last': False,
                   'collate_fn': mpnn_collate_func,
                   'sampler': None}

    training_generator = DataLoader(
        data_process_loader(train_df.index.values, train_df.Label.values, train_df, **config),
        **params)

    dev_generator = DataLoader(
        data_process_loader(dev_df.index.values, dev_df.Label.values, dev_df, **config),
        **{**params_test, 'sampler': SequentialSampler(
            data_process_loader(dev_df.index.values, dev_df.Label.values, dev_df, **config))})

    test_generator = DataLoader(
        data_process_loader(test_df.index.values, test_df.Label.values, test_df, **config),
        **{**params_test, 'sampler': SequentialSampler(
            data_process_loader(test_df.index.values, test_df.Label.values, test_df, **config))})

    model.binary = True
    opt = torch.optim.Adam(model.model.parameters(), lr=config['LR'], weight_decay=config.get('decay', 0))
    model_max = copy.deepcopy(model.model)

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        n_batches = 0
        for i, (v_d, v_p, label) in enumerate(training_generator):
            if model.target_encoding == 'Transformer':
                v_p = v_p
            else:
                v_p = v_p.float().to(device)
            if model.drug_encoding in ["MPNN"]:
                v_d = v_d
            else:
                v_d = v_d.float().to(device)

            score = model.model(v_d, v_p)
            label = torch.autograd.Variable(torch.from_numpy(np.array(label)).float()).to(device)
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score), 1)
            loss = loss_fct(n, label)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        all_split_metrics = {}
        for split_name, gen in [('train', training_generator), ('dev', dev_generator), ('test', test_generator)]:
            model.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for v_d, v_p, label in gen:
                    if model.target_encoding == 'Transformer':
                        v_p = v_p
                    else:
                        v_p = v_p.float().to(device)
                    if model.drug_encoding in ["MPNN"]:
                        v_d = v_d
                    else:
                        v_d = v_d.float().to(device)
                    score = model.model(v_d, v_p)
                    m = torch.nn.Sigmoid()
                    logits = torch.squeeze(m(score)).detach().cpu().numpy()
                    label_ids = label
                    y_label = label_ids.flatten().tolist()
                    y_pred = logits.flatten().tolist()
                    all_preds.extend(y_pred)
                    all_labels.extend(y_label)

            opt_thresh = find_optimal_threshold(np.array(all_labels), np.array(all_preds))
            metrics = compute_metrics(all_labels, all_preds, opt_thresh)
            metrics['Threshold'] = opt_thresh
            all_split_metrics[split_name] = metrics

        log_entry = {
            'epoch': epoch + 1, 'train_loss': avg_loss,
            'train': {k: v for k, v in all_split_metrics['train'].items()},
            'dev': {k: v for k, v in all_split_metrics['dev'].items()},
            'test': {k: v for k, v in all_split_metrics['test'].items()},
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        dev_metrics = all_split_metrics['dev']
        test_metrics = all_split_metrics['test']
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | "
              f"Dev AUC={dev_metrics['AUC']:.4f} F1={dev_metrics['F1']:.4f} "
              f"Acc={dev_metrics['Accuracy']:.4f} | "
              f"Test AUC={test_metrics['AUC']:.4f}")

        if dev_metrics['AUC'] > best_auc:
            best_auc = dev_metrics['AUC']
            patience_counter = 0
            model_max = copy.deepcopy(model.model)
            torch.save({'epoch': epoch, 'model_state_dict': model_max.state_dict(),
                        'best_auc': best_auc, 'metrics': dev_metrics},
                       os.path.join(LOG_DIR, 'best_model.pt'))
            print(f"  *** Best model saved (Dev AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nDone. Best Dev AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
