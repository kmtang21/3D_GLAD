"""
Generate ablation study tables.
Reads test_results_<model>.csv from Ablations/<group>/ and 3DGLAD/test_results.csv,
computes AUC, AUPRC, F1, Accuracy, Recall, Specificity, and outputs grouped tables.

Produces:
  - results/ablation_encoder_table.csv  (encoder ablation: GCN vs TAG vs Transformer vs GIN vs GATv2)
  - results/ablation_decoder_table.csv  (decoder ablation: CrossAttn vs NTN vs MLP vs NodeBilinear vs Bilinear)
  - results/ablation_gae_table.csv      (GAE ablation: with GAE vs without GAE)
  - results/ablation_all_table.csv      (combined)
  - corresponding .tex files
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, recall_score,
                             confusion_matrix, accuracy_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ABL_DIR = os.path.join(PROJECT_DIR, 'Ablations')
GLAD_DIR = os.path.join(PROJECT_DIR, '3DGLAD')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

METRIC_COLS = ['AUC', 'AUPRC', 'F1', 'Accuracy', 'Recall', 'Specificity']


def find_optimal_threshold(labels, preds):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1v = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_t = f1v, t
    return best_t


def compute_full_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc_val = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    opt_thresh = find_optimal_threshold(y_true, y_pred)
    pred_binary = (y_pred >= opt_thresh).astype(int)
    f1 = f1_score(y_true, pred_binary, zero_division=0)
    rec = recall_score(y_true, pred_binary, zero_division=0)
    acc = accuracy_score(y_true, pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {c: v for c, v in zip(METRIC_COLS, [auc_val, auprc, f1, acc, rec, spec])}


def load_csv(path):
    df = pd.read_csv(path)
    return df['Label'].values, df['Prediction'].values


def load_model(name, csv_path):
    if not os.path.exists(csv_path):
        return None
    y_true, y_pred = load_csv(csv_path)
    metrics = compute_full_metrics(y_true, y_pred)
    metrics['Model'] = name
    return metrics


ABLATION_GROUPS = {
    'encoder': {
        'models': [
            ('3DGLAD (GCN+CrossAttn)', os.path.join(GLAD_DIR, 'test_results.csv')),
            ('abl_encoder_TAG',         os.path.join(ABL_DIR, 'encoder', 'test_results_abl_encoder_TAG.csv')),
            ('abl_encoder_Transformer', os.path.join(ABL_DIR, 'encoder', 'test_results_abl_encoder_Transformer.csv')),
            ('abl_encoder_GIN',         os.path.join(ABL_DIR, 'encoder', 'test_results_abl_encoder_GIN.csv')),
            ('abl_encoder_GATv2',       os.path.join(ABL_DIR, 'encoder', 'test_results_abl_encoder_GATv2.csv')),
        ],
        'display': {
            '3DGLAD (GCN+CrossAttn)': 'GCN + CrossAttn (Ours)',
            'abl_encoder_TAG':         'TAG + CrossAttn',
            'abl_encoder_Transformer': 'Transformer + CrossAttn',
            'abl_encoder_GIN':         'GIN + CrossAttn',
            'abl_encoder_GATv2':       'GATv2 + CrossAttn',
        },
    },
    'decoder': {
        'models': [
            ('3DGLAD (GCN+CrossAttn)', os.path.join(GLAD_DIR, 'test_results.csv')),
            ('abl_decoder_ntn',          os.path.join(ABL_DIR, 'decoder', 'test_results_abl_decoder_ntn.csv')),
            ('abl_decoder_mlp',          os.path.join(ABL_DIR, 'decoder', 'test_results_abl_decoder_mlp.csv')),
            ('abl_decoder_nodebilinear', os.path.join(ABL_DIR, 'decoder', 'test_results_abl_decoder_nodebilinear.csv')),
            ('abl_decoder_bilinear',     os.path.join(ABL_DIR, 'decoder', 'test_results_abl_decoder_bilinear.csv')),
        ],
        'display': {
            '3DGLAD (GCN+CrossAttn)': 'GCN + CrossAttn (Ours)',
            'abl_decoder_ntn':          'GCN + NTN',
            'abl_decoder_mlp':          'GCN + MLP',
            'abl_decoder_nodebilinear': 'GCN + NodeBilinear',
            'abl_decoder_bilinear':     'GCN + Bilinear',
        },
    },
    'gae': {
        'models': [
            ('abl_withGAE',    os.path.join(ABL_DIR, 'withoutGAE', 'test_results_abl_withGAE.csv')),
            ('abl_withoutGAE', os.path.join(ABL_DIR, 'withoutGAE', 'test_results_abl_withoutGAE.csv')),
        ],
        'display': {
            'abl_withGAE':    'GCN + CrossAttn + GAE (Ours)',
            'abl_withoutGAE': 'GCN + CrossAttn (no GAE)',
        },
    },
}


def bold_best_str(val, best_val):
    if val == best_val:
        return f'\\textbf{{{val:.4f}}}'
    return f'{val:.4f}'


def build_table(rows_data, display_map):
    rows = []
    for raw_name, metrics in rows_data:
        display = display_map.get(raw_name, raw_name)
        row = {'Model': display}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    df = df[METRIC_COLS]
    return df


def save_csv_tex(df, prefix):
    csv_path = os.path.join(RESULTS_DIR, f'{prefix}.csv')
    tex_path = os.path.join(RESULTS_DIR, f'{prefix}.tex')

    df.to_csv(csv_path, float_format='%.4f')

    nice_name = prefix.replace('ablation_', '').capitalize()
    if nice_name == 'All':
        nice_name = 'All ablation variants'
    elif nice_name == 'Gae':
        nice_name = 'GAE'

    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Ablation study: {nice_name}}}\n')
        f.write(f'\\label{{tab:ablation_{prefix}}}\n')
        f.write('\\begin{tabular}{l' + 'c' * len(METRIC_COLS) + '}\n')
        f.write('\\toprule\n')
        f.write('Model & ' + ' & '.join(METRIC_COLS) + ' \\\\\n')
        f.write('\\midrule\n')
        for model_name in df.index:
            vals = []
            for col in METRIC_COLS:
                v = float(df.loc[model_name, col])
                best = float(df[col].max())
                vals.append(bold_best_str(v, best))
            f.write(f"{model_name} & {' & '.join(vals)} \\\\\n")
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {tex_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate ablation study tables')
    parser.add_argument('--group', choices=['encoder', 'decoder', 'gae', 'all'], default='all')
    args = parser.parse_args()

    groups_to_run = [args.group] if args.group != 'all' else ['encoder', 'decoder', 'gae']

    all_rows = []
    for group_name in groups_to_run:
        cfg = ABLATION_GROUPS[group_name]
        print(f"\n{'='*60}")
        print(f" Ablation: {group_name}")
        print(f"{'='*60}")

        rows_data = []
        for raw_name, csv_path in cfg['models']:
            metrics = load_model(raw_name, csv_path)
            if metrics is not None:
                del metrics['Model']
                rows_data.append((raw_name, metrics))
                print(f"  {raw_name}: AUC={metrics['AUC']:.4f}")
            else:
                print(f"  {raw_name}: CSV not found at {csv_path}")

        if not rows_data:
            print("  No data, skipping")
            continue

        df = build_table(rows_data, cfg['display'])
        df = df.sort_values('AUC', ascending=False)
        save_csv_tex(df, f'ablation_{group_name}')

        print(f"\n{df.to_string(float_format='%.4f')}")

        all_rows.extend(rows_data)

    if args.group == 'all' and all_rows:
        print(f"\n{'='*60}")
        print(f" Ablation: ALL")
        print(f"{'='*60}")

        glad_name = '3DGLAD (GCN+CrossAttn)'
        gae_name = 'abl_withGAE'
        glad_metrics = None
        seen = set()
        unique_rows = []
        for raw_name, metrics in all_rows:
            if raw_name == glad_name:
                if glad_metrics is None:
                    glad_metrics = (raw_name, metrics)
                continue
            if raw_name == gae_name:
                continue
            if raw_name not in seen:
                seen.add(raw_name)
                unique_rows.append((raw_name, metrics))
        unique_rows.insert(0, glad_metrics)

        all_display = {}
        for g in ABLATION_GROUPS.values():
            all_display.update(g['display'])
        all_display[glad_name] = '3DGLAD (Ours)'

        df_all = build_table(unique_rows, all_display)
        df_all = df_all.sort_values('AUC', ascending=False)
        save_csv_tex(df_all, 'ablation_all')
        print(f"\n{df_all.to_string(float_format='%.4f')}")

    print("\nDone!")


if __name__ == '__main__':
    main()
