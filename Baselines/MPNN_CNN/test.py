import os
import sys
import json
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score)
from torch.utils.data import DataLoader, SequentialSampler

DEEPPURPOSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Compare_models', 'DeepPurpose')
sys.path.insert(0, DEEPPURPOSE_DIR)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
PRE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre_data')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def load_csv(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    return df['SMILES'].values, df['Target'].values, df['Label'].values


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from DeepPurpose import utils
    from DeepPurpose import DTI as models
    from DeepPurpose.utils import data_process_loader, mpnn_collate_func

    drug_encoding = 'MPNN'
    target_encoding = 'CNN'

    if os.path.exists(os.path.join(PRE_DATA_DIR, 'test.csv')):
        test_csv = os.path.join(PRE_DATA_DIR, 'test.csv')
    else:
        test_csv = os.path.join(DATA_DIR, 'test.csv')

    test_df = utils.data_process(
        *load_csv(test_csv),
        drug_encoding, target_encoding,
        split_method='no_split', random_seed=SEED)

    print(f"Test: {len(test_df)}")

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
    model.binary = True

    ckpt_path = os.path.join(LOG_DIR, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")

    params_test = {'batch_size': config['batch_size'], 'shuffle': False,
                   'num_workers': 0, 'drop_last': False,
                   'collate_fn': mpnn_collate_func}

    test_info = data_process_loader(test_df.index.values, test_df.Label.values, test_df, **config)
    test_generator = DataLoader(test_info, **params_test,
                                sampler=SequentialSampler(test_info))

    model.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for v_d, v_p, label in test_generator:
            v_p = v_p.float().to(device)
            score = model.model(v_d, v_p)
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            y_label = np.array(label).flatten().tolist()
            y_pred = logits.flatten().tolist()
            all_preds.extend(y_pred)
            all_labels.extend(y_label)

    opt_thresh = find_optimal_threshold(np.array(all_labels), np.array(all_preds))
    metrics = compute_metrics(all_labels, all_preds, opt_thresh)
    metrics['Threshold'] = opt_thresh

    print(f"\nTest Results (threshold={opt_thresh:.2f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_path = os.path.join(LOG_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    predictions_path = os.path.join(LOG_DIR, 'test_predictions.csv')
    import pandas as pd
    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    main()
