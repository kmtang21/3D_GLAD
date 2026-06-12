"""
Plot ROC and PRC curves for all baselines + 3DGLAD.
Reads test_results.csv (Label, Prediction) from each baseline directory
and the 3DGLAD predictions from results/test_predictions/3DGLAD.pkl.
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_DIRS = {
    'TransformerCPI': os.path.join(PROJECT_DIR, 'Baselines', 'TransformerCPI'),
    'CPI_prediction': os.path.join(PROJECT_DIR, 'Baselines', 'CPI_prediction'),
    'MPNN_CNN': os.path.join(PROJECT_DIR, 'Baselines', 'MPNN_CNN'),
    'MSFF_DTA': os.path.join(PROJECT_DIR, 'Baselines', 'MSFF_DTA'),
    'AttentionSiteDTI': os.path.join(PROJECT_DIR, 'Baselines', 'AttentionSiteDTI'),
}

DISPLAY_NAMES = {
    '3DGLAD': '3DGLAD (Ours)',
    'TransformerCPI': 'TransformerCPI',
    'CPI_prediction': 'CPI_prediction',
    'MPNN_CNN': 'MPNN_CNN',
    'MSFF_DTA': 'MSFF_DTA',
    'AttentionSiteDTI': 'AttentionSiteDTI',
}

COLORS = {
    '3DGLAD': '#E74C3C',
    'TransformerCPI': '#3498DB',
    'CPI_prediction': '#2ECC71',
    'MPNN_CNN': '#F39C12',
    'MSFF_DTA': '#9B59B6',
    'AttentionSiteDTI': '#1ABC9C',
}

MODEL_ORDER = [
    '3DGLAD',
    'TransformerCPI',
    'MSFF_DTA',
    'CPI_prediction',
    'MPNN_CNN',
    'AttentionSiteDTI',
]


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


def load_all_predictions():
    data = {}
    y_true, y_pred = load_3dglad_predictions()
    if y_true is not None:
        data['3DGLAD'] = (y_true, y_pred)

    for name, model_dir in BASELINE_DIRS.items():
        y_true, y_pred = load_baseline_predictions(name, model_dir)
        if y_true is not None:
            data[name] = (y_true, y_pred)
        else:
            print(f"  Warning: no test_results.csv for {name}")

    return data


def plot_roc_prc(data, output_prefix='baselines'):
    models_in_data = [m for m in MODEL_ORDER if m in data]
    if not models_in_data:
        print("  No data to plot")
        return

    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 7))
    fig_prc, ax_prc = plt.subplots(1, 1, figsize=(8, 7))

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    for name in models_in_data:
        y_true, y_pred = data[name]
        display = DISPLAY_NAMES.get(name, name)
        color = COLORS.get(name, None)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc_val = auc(fpr, tpr)
        lw = 2.5 if name == '3DGLAD' else 1.5
        ax_roc.plot(fpr, tpr, color=color, lw=lw,
                    label=f'{display} (AUC={roc_auc_val:.4f})')

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        ax_prc.plot(recall, precision, color=color, lw=lw,
                    label=f'{display} (AP={ap:.4f})')

    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve - Baselines vs 3DGLAD', fontsize=14)
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(True, alpha=0.3)

    ax_prc.set_xlim([0, 1])
    ax_prc.set_ylim([0, 1.02])
    ax_prc.set_xlabel('Recall', fontsize=12)
    ax_prc.set_ylabel('Precision', fontsize=12)
    ax_prc.set_title('Precision-Recall Curve - Baselines vs 3DGLAD', fontsize=14)
    ax_prc.legend(loc='lower left', fontsize=10)
    ax_prc.grid(True, alpha=0.3)

    fig_roc.tight_layout()
    fig_prc.tight_layout()

    roc_path = os.path.join(OUT_DIR, f'roc_{output_prefix}.png')
    prc_path = os.path.join(OUT_DIR, f'prc_{output_prefix}.png')
    fig_roc.savefig(roc_path, dpi=200)
    fig_prc.savefig(prc_path, dpi=200)
    plt.close(fig_roc)
    plt.close(fig_prc)
    print(f"  Saved: {roc_path}")
    print(f"  Saved: {prc_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot ROC and PRC for baselines + 3DGLAD')
    parser.add_argument('--output_prefix', type=str, default='baselines',
                        help='Prefix for output filenames')
    args = parser.parse_args()

    print("Loading predictions...")
    data = load_all_predictions()
    print(f"  Found {len(data)} models: {list(data.keys())}")

    print("Plotting...")
    plot_roc_prc(data, args.output_prefix)
    print("Done!")


if __name__ == '__main__':
    main()
