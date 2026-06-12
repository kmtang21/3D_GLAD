"""
Generate a summary table from test_results.csv files of all baselines + 3DGLAD.
Reads test_results.csv (Label, Prediction) from each baseline directory and
results/test_predictions/3DGLAD.pkl, computes metrics, and outputs a LaTeX-ready table.

Output columns: Model, AUC, AUPRC, F1, Accuracy, Recall, Specificity
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, accuracy_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

BASELINE_DIRS = {
    'TransformerCPI': os.path.join(PROJECT_DIR, 'Baselines', 'TransformerCPI'),
    'CPI_prediction': os.path.join(PROJECT_DIR, 'Baselines', 'CPI_prediction'),
    'MPNN_CNN': os.path.join(PROJECT_DIR, 'Baselines', 'MPNN_CNN'),
    'MSFF_DTA': os.path.join(PROJECT_DIR, 'Baselines', 'MSFF_DTA'),
    'AttentionSiteDTI': os.path.join(PROJECT_DIR, 'Baselines', 'AttentionSiteDTI'),
}

MODEL_ORDER = [
    '3DGLAD',
    'TransformerCPI',
    'MSFF_DTA',
    'CPI_prediction',
    'MPNN_CNN',
    'AttentionSiteDTI',
]

DISPLAY_NAMES = {
    '3DGLAD': '3DGLAD (Ours)',
    'TransformerCPI': 'TransformerCPI',
    'CPI_prediction': 'CPI_prediction',
    'MPNN_CNN': 'MPNN-CNN',
    'MSFF_DTA': 'MSFF-DTA',
    'AttentionSiteDTI': 'AttentionSiteDTI',
}

METRIC_COLS = ['AUC', 'AUPRC', 'F1', 'Accuracy', 'Recall', 'Specificity']


def find_optimal_threshold(labels, preds):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1v = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_t = f1v, t
    return best_t


def compute_full_metrics(y_true, y_pred_scores):
    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    auc_val = roc_auc_score(y_true, y_pred_scores)
    auprc = average_precision_score(y_true, y_pred_scores)
    opt_thresh = find_optimal_threshold(y_true, y_pred_scores)
    pred_binary = (y_pred_scores >= opt_thresh).astype(int)
    f1 = f1_score(y_true, pred_binary, zero_division=0)
    rec = recall_score(y_true, pred_binary, zero_division=0)
    acc = accuracy_score(y_true, pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'AUC': auc_val,
        'AUPRC': auprc,
        'F1': f1,
        'Accuracy': acc,
        'Recall': rec,
        'Specificity': spec,
    }


def load_3dglad_predictions():
    csv_path = os.path.join(PROJECT_DIR, '3D_GLAD', 'test_results.csv')
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if 'Label' not in df.columns or 'Prediction' not in df.columns:
        return None, None
    return df['Label'].values, df['Prediction'].values


def load_baseline_predictions(model_name, model_dir):
    csv_path = os.path.join(model_dir, 'test_results.csv')
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if 'Label' not in df.columns or 'Prediction' not in df.columns:
        return None, None
    return df['Label'].values, df['Prediction'].values


def bold_best(df, metric_cols):
    result = df.copy()
    for col in metric_cols:
        best_val = float(result[col].max())
        formatted = []
        for idx in result.index:
            val = float(result.loc[idx, col])
            if val == best_val:
                formatted.append(f'\\textbf{{{val:.4f}}}')
            else:
                formatted.append(f'{val:.4f}')
        result[col] = formatted
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate summary table for baselines + 3DGLAD')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: results/baseline_table.csv)')
    parser.add_argument('--latex', action='store_true',
                        help='Also output LaTeX formatted table')
    args = parser.parse_args()

    all_metrics = {}

    y_true, y_pred = load_3dglad_predictions()
    if y_true is not None:
        all_metrics['3DGLAD'] = compute_full_metrics(y_true, y_pred)
        print(f"  3DGLAD: AUC={all_metrics['3DGLAD']['AUC']:.4f}")
    else:
        print("  Warning: 3DGLAD predictions not found")

    for name, model_dir in BASELINE_DIRS.items():
        y_true, y_pred = load_baseline_predictions(name, model_dir)
        if y_true is not None:
            all_metrics[name] = compute_full_metrics(y_true, y_pred)
            print(f"  {name}: AUC={all_metrics[name]['AUC']:.4f}")
        else:
            print(f"  Warning: no test_results.csv for {name}")

    rows = []
    for name in MODEL_ORDER:
        if name in all_metrics:
            row = {'Model': DISPLAY_NAMES.get(name, name)}
            row.update(all_metrics[name])
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('Model')

    numeric_df = df.copy()
    for col in METRIC_COLS:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    df_sorted = numeric_df.sort_values('AUC', ascending=False)

    output_path = args.output or os.path.join(PROJECT_DIR, 'results', 'baseline_table.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sorted.to_csv(output_path, float_format='%.4f')

    print(f"\n{'='*80}")
    print("BASELINE COMPARISON TABLE (sorted by AUC)")
    print(f"{'='*80}")
    print(df_sorted.to_string(float_format='%.4f'))
    print(f"\nSaved to: {output_path}")

    if args.latex:
        latex_df = df_sorted.reset_index()
        latex_df = bold_best(latex_df.copy(), METRIC_COLS)
        latex_path = output_path.replace('.csv', '.tex')
        with open(latex_path, 'w') as f:
            f.write('\\begin{table}[htbp]\n')
            f.write('\\centering\n')
            f.write('\\caption{Performance comparison of baselines and 3DGLAD on the test set}\n')
            f.write('\\label{tab:baseline_comparison}\n')
            f.write('\\begin{tabular}{l' + 'c' * len(METRIC_COLS) + '}\n')
            f.write('\\toprule\n')
            f.write('Model & ' + ' & '.join(METRIC_COLS) + ' \\\\\n')
            f.write('\\midrule\n')
            for _, row in latex_df.iterrows():
                vals = ' & '.join(str(row[c]) for c in METRIC_COLS)
                f.write(f"{row['Model']} & {vals} \\\\\n")
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n')
        print(f"LaTeX table saved to: {latex_path}")


if __name__ == '__main__':
    main()
