"""
Run all baseline models on the InterpretableDTIP test set.
Saves per-model test_predictions.csv (Label, Prediction) in each model's subdirectory.
Also computes and prints standard metrics.
"""
import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, accuracy_score)
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['RDKIT_ERROR_LEVEL'] = 'ERROR'
os.environ['RDKIT_WARNINGS_SILENCE'] = '1'
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


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
    prec = precision_score(y_true, pred_binary, zero_division=0)
    rec = recall_score(y_true, pred_binary, zero_division=0)
    acc = accuracy_score(y_true, pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'AUC': round(auc_val, 4),
        'AUPRC': round(auprc, 4),
        'F1': round(f1, 4),
        'Accuracy': round(acc, 4),
        'Recall': round(rec, 4),
        'Specificity': round(spec, 4),
        'Threshold': round(opt_thresh, 2),
    }


# ==============================================================
# Data loading helpers (shared across GNN-based baselines)
# ==============================================================
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"

sys.path.insert(0, BASE_DIR)
from data_loader import (load_all_data, build_unified_maps,
                          smiles_to_atom_features_adj,
                          protein_seq_to_onehot,
                          AtomFeatureDataset)
from torch.utils.data import DataLoader


def get_test_data():
    data_dir = os.path.join(PROJECT_DIR, 'InterpretableDTIP', 'data')
    train_data, dev_data, test_data = load_all_data(data_dir)
    unified_smiles, unified_seqs, unified_c2i, unified_p2i = \
        build_unified_maps(train_data, dev_data, test_data)
    return test_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i


def precompute_features(unified_smiles, unified_seqs, max_atoms=100, max_protein_len=1000):
    drug_feats = np.zeros((len(unified_smiles), max_atoms, 26), dtype=np.float32)
    drug_adjs = np.zeros((len(unified_smiles), max_atoms, max_atoms), dtype=np.float32)
    drug_atom_nums = np.zeros(len(unified_smiles), dtype=np.int64)
    for i, smi in enumerate(tqdm(unified_smiles, desc="Drug features")):
        feat, adj, n = smiles_to_atom_features_adj(smi, max_atoms)
        if feat is not None:
            drug_feats[i] = feat
            drug_adjs[i] = adj
            drug_atom_nums[i] = n

    protein_onehots = np.zeros((len(unified_seqs), max_protein_len, 22), dtype=np.float32)
    protein_lens = np.zeros(len(unified_seqs), dtype=np.int64)
    for i, seq in enumerate(tqdm(unified_seqs, desc="Protein features")):
        oh, slen = protein_seq_to_onehot(seq, max_protein_len)
        protein_onehots[i] = oh
        protein_lens[i] = slen

    return drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens


# ==============================================================
# CPI_prediction
# ==============================================================
def test_cpi_prediction(device):
    from train_cpi_prediction import CPIPredictionModel

    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'CPI_prediction')
    out_dir = os.path.join(model_dir, 'logs')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(model_dir, 'test_results.csv')

    if os.path.exists(csv_path):
        print(f"  [CPI_prediction] test_results.csv already exists, skipping")
        return

    test_data, unified_smiles, unified_seqs, unified_c2i, unified_p2i = get_test_data()
    drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens = \
        precompute_features(unified_smiles, unified_seqs)

    test_dataset = AtomFeatureDataset(
        drug_feats, drug_adjs, drug_atom_nums, protein_onehots, protein_lens,
        unified_c2i, unified_p2i, test_data[4], test_data[5])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CPIPredictionModel().to(device)
    ckpt_path = os.path.join(model_dir, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feat, adj, atom_num, protein_oh, protein_len, label in tqdm(test_loader, desc="CPI_prediction"):
            feat, adj = feat.to(device), adj.to(device)
            protein_oh = protein_oh.to(device)
            for i in range(feat.size(0)):
                out = model(feat[i], adj[i], atom_num[i], protein_oh[i], protein_len[i])
                prob = F.softmax(out, dim=1)[0, 1].item()
                all_preds.append(prob)
                all_labels.append(label[i].item())

    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(csv_path, index=False)
    metrics = compute_full_metrics(all_labels, all_preds)
    print(f"  AUC={metrics['AUC']}, AUPRC={metrics['AUPRC']}, F1={metrics['F1']}")
    return metrics


# ==============================================================
# TransformerCPI
# ==============================================================
def test_transformer_cpi(device):
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'TransformerCPI')
    csv_path = os.path.join(model_dir, 'test_results.csv')

    if os.path.exists(csv_path):
        print(f"  [TransformerCPI] test_results.csv already exists, skipping")
        return

    sys.path.insert(0, model_dir)
    from model import (Encoder, Decoder, Predictor, pack,
                        DecoderLayer, SelfAttention, PositionwiseFeedforward)

    test_pkl = None
    for d in [os.path.join(model_dir, 'pre_data'), os.path.join(model_dir, 'data')]:
        p = os.path.join(d, 'test.pkl')
        if os.path.exists(p):
            test_pkl = p
            break

    if test_pkl is None:
        print(f"  [TransformerCPI] No test.pkl found, skipping")
        return

    with open(test_pkl, 'rb') as f:
        dataset_test = pickle.load(f)
    print(f"  Test samples: {len(dataset_test)}")

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

    ckpt_path = os.path.join(model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset_test)), desc="TransformerCPI"):
            atom, adj, protein, label = dataset_test[idx]
            adjs, atoms, proteins, labels_list = [adj], [atom], [protein], [label]
            data_pack = pack(atoms, adjs, proteins, labels_list, device)
            _, _, scores = model(data_pack, train=False)
            all_preds.append(scores[0])
            all_labels.append(label if isinstance(label, int) else label.item())

    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(csv_path, index=False)
    metrics = compute_full_metrics(all_labels, all_preds)
    print(f"  AUC={metrics['AUC']}, AUPRC={metrics['AUPRC']}, F1={metrics['F1']}")
    return metrics


# ==============================================================
# MPNN_CNN (DeepPurpose)
# ==============================================================
def test_mpnn_cnn(device):
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'MPNN_CNN')
    csv_path = os.path.join(model_dir, 'test_results.csv')

    if os.path.exists(csv_path):
        print(f"  [MPNN_CNN] test_results.csv already exists, skipping")
        return

    DP_DIR = os.path.join(PROJECT_DIR, 'Compare_models', 'DeepPurpose')
    sys.path.insert(0, DP_DIR)
    sys.path.insert(0, model_dir)

    from DeepPurpose import utils
    from DeepPurpose import DTI as models
    from DeepPurpose.utils import data_process_loader, mpnn_collate_func
    from torch.utils.data import SequentialSampler

    pre_data_dir = os.path.join(model_dir, 'pre_data')
    data_subdir = os.path.join(model_dir, 'data')
    DATA_DIR = pre_data_dir if os.path.exists(pre_data_dir) else data_subdir

    def load_csv(filepath):
        df = pd.read_csv(filepath)
        return df['SMILES'].values, df['Target'].values, df['Label'].values

    test_csv = os.path.join(DATA_DIR, 'test.csv')
    test_df = utils.data_process(
        *load_csv(test_csv), 'MPNN', 'CNN',
        split_method='no_split', random_seed=42)

    config = utils.generate_config(
        drug_encoding='MPNN', target_encoding='CNN',
        result_folder=os.path.join(model_dir, 'logs', 'dp_results'),
        cls_hidden_dims=[1024, 1024, 512], train_epoch=100, LR=0.001,
        batch_size=256, hidden_dim_drug=128, mpnn_hidden_size=128, mpnn_depth=3,
        cnn_target_filters=[32, 64, 64], cnn_target_kernels=[4, 8, 8],
        num_workers=0, cuda_id=None)
    config['decay'] = 0

    model = models.model_initialize(**config)
    model.device = device
    model.model = model.model.to(device)
    model.binary = True

    ckpt_path = os.path.join(model_dir, 'best_model.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])

    test_info = data_process_loader(test_df.index.values, test_df.Label.values, test_df, **config)
    test_generator = DataLoader(test_info, batch_size=config['batch_size'], shuffle=False,
                                num_workers=0, drop_last=False, collate_fn=mpnn_collate_func,
                                sampler=SequentialSampler(test_info))

    model.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for v_d, v_p, label in tqdm(test_generator, desc="MPNN_CNN"):
            v_p = v_p.float().to(device)
            score = model.model(v_d, v_p)
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            all_preds.extend(logits.flatten().tolist())
            all_labels.extend(np.array(label).flatten().tolist())

    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(csv_path, index=False)
    metrics = compute_full_metrics(all_labels, all_preds)
    print(f"  AUC={metrics['AUC']}, AUPRC={metrics['AUPRC']}, F1={metrics['F1']}")
    return metrics


# ==============================================================
# MSFF_DTA
# ==============================================================
def test_msff_dta(device):
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'MSFF_DTA')
    csv_path = os.path.join(model_dir, 'test_results.csv')

    if os.path.exists(csv_path):
        print(f"  [MSFF_DTA] test_results.csv already exists, skipping")
        return

    MSFF_DIR = os.path.join(PROJECT_DIR, 'Compare_models', 'MSFF-DTA')
    sys.path.insert(0, MSFF_DIR)
    sys.path.insert(0, model_dir)

    from preprocessing.protein import ProteinFeatureManager
    from preprocessing.compound import CompoundFeatureManager
    from data import CPIDataset
    from core import Predictor
    from data_preprocess import set_seed, SEED
    from torch.utils.data import DataLoader

    set_seed(SEED)

    pre_data_dir = os.path.join(model_dir, 'pre_data')
    msff_data_dir = os.path.join(model_dir, 'msff_data')
    DATA_DIR = pre_data_dir if os.path.exists(pre_data_dir) else msff_data_dir

    protein_fm = ProteinFeatureManager(DATA_DIR)
    compound_fm = CompoundFeatureManager(DATA_DIR)

    model_args = argparse.Namespace(
        objective='classification', batch_size=64, num_workers=0,
        learning_rate=4e-6, max_epochs=100, early_stop_round=20,
        root_data_path=DATA_DIR, decoder_layers=3, n_heads=3, gnn_layers=3,
        protein_gnn_dim=64, compound_gnn_dim=34, mol2vec_embedding_dim=300,
        hid_dim=64, pf_dim=256, dropout=0.2, protein_encoder_layers=3,
        protein_encoder_head=4, cnn_kernel_size=7, protein_dim=64, atom_dim=34,
        edge_dim=6, protein_embedding_dim=1280, compound_embedding_dim=2727, seed=SEED)

    test_csv = os.path.join(DATA_DIR, 'test.csv')
    test_dataset = CPIDataset(test_csv, protein_fm, compound_fm, model_args)
    test_loader = DataLoader(test_dataset, batch_size=model_args.batch_size,
                             collate_fn=test_dataset.collate_fn, shuffle=False, num_workers=0)

    model = Predictor(model_args).to(device)
    ckpt_path = os.path.join(model_dir, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MSFF_DTA"):
            labels_t = batch["LABEL"].long()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif hasattr(v, 'to'):
                    batch[k] = v.to(device)
            outputs, _, _ = model(batch)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist()
            all_preds.extend(scores)
            all_labels.extend(labels_t.numpy().tolist())

    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(csv_path, index=False)
    metrics = compute_full_metrics(all_labels, all_preds)
    print(f"  AUC={metrics['AUC']}, AUPRC={metrics['AUPRC']}, F1={metrics['F1']}")
    return metrics


# ==============================================================
# AttentionSiteDTI
# ==============================================================
def test_attentionsitedti(device):
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'AttentionSiteDTI')
    csv_path = os.path.join(model_dir, 'test_results.csv')

    if os.path.exists(csv_path):
        print(f"  [AttentionSiteDTI] test_results.csv already exists, skipping")
        return

    sys.path.insert(0, model_dir)
    from layers import MultiHeadAttention
    from torch_geometric.nn import TAGConv, GlobalAttention

    class DTITAG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.protein_graph_conv = torch.nn.ModuleList([TAGConv(31, 31, K=2) for _ in range(5)])
            self.ligand_graph_conv = torch.nn.ModuleList([
                TAGConv(74, 70, K=2), TAGConv(70, 65, K=2), TAGConv(65, 60, K=2),
                TAGConv(60, 55, K=2), TAGConv(55, 31, K=2)])
            self.pooling_protein = GlobalAttention(torch.nn.Linear(31, 1))
            self.pooling_ligand = GlobalAttention(torch.nn.Linear(31, 1))
            self.dropout_rate = 0.2
            self.bilstm = torch.nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=0.0)
            self.fc_in = torch.nn.Linear(8680, 4340)
            self.fc_out = torch.nn.Linear(4340, 1)
            self.attention = MultiHeadAttention(62, 62, 2)

        def forward(self, protein_graph, drug_graph, device):
            feature_protein = protein_graph.x
            feature_smile = drug_graph.x
            protein_edge_index = protein_graph.edge_index
            drug_edge_index = drug_graph.edge_index
            for module in self.protein_graph_conv:
                feature_protein = F.relu(module(feature_protein, protein_edge_index))
            for module in self.ligand_graph_conv:
                feature_smile = F.relu(module(feature_smile, drug_edge_index))
            protein_reps = self.pooling_protein(feature_protein, protein_graph.batch).view(-1, 31)
            drug_batch = torch.zeros(feature_smile.size(0), dtype=torch.long, device=device)
            ligand_rep = self.pooling_ligand(feature_smile, drug_batch).view(-1, 31)
            sequence = torch.cat((ligand_rep, protein_reps), dim=0).view(1, -1, 31)
            seq_len = sequence.size(1)
            mask = torch.eye(140, dtype=torch.bool).view(1, 140, 140).to(device)
            mask[0, seq_len:140, :] = False
            mask[0, :, seq_len:140] = False
            mask[0, :, seq_len - 1] = True
            mask[0, seq_len - 1, :] = True
            mask[0, seq_len - 1, seq_len - 1] = False
            sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - seq_len), mode='constant', value=0)
            sequence = sequence.permute(1, 0, 2)
            h_0 = torch.zeros(2, 1, 31).to(device)
            c_0 = torch.zeros(2, 1, 31).to(device)
            output, _ = self.bilstm(sequence, (h_0, c_0))
            output = output.permute(1, 0, 2)
            out = self.attention(output, mask=mask)
            out = F.relu(self.fc_in(out.view(-1, out.size(1) * out.size(2))))
            out = torch.sigmoid(self.fc_out(out))
            return out

    cache_file = os.path.join(model_dir, 'pre_data', 'attention_data.pkl')
    if not os.path.exists(cache_file):
        cache_file = os.path.join(PROJECT_DIR, '3D_GLAD', 'cache', 'attention_data.pkl')
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    test_ds = data['test']

    ckpt_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, 'logs', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_DIR, 'AttentionSiteDTI', 'interpretable_dtip_output', 'best_model.pt')
    model = DTITAG().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="AttentionSiteDTI"):
            pgraph, dgraph, label = test_ds[i]
            pgraph, dgraph = pgraph.to(device), dgraph.to(device)
            out = model(pgraph, dgraph, device)
            all_preds.append(out.cpu().item())
            all_labels.append(label)

    pd.DataFrame({'Label': all_labels, 'Prediction': all_preds}).to_csv(csv_path, index=False)
    metrics = compute_full_metrics(all_labels, all_preds)
    print(f"  AUC={metrics['AUC']}, AUPRC={metrics['AUPRC']}, F1={metrics['F1']}")
    return metrics


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description='Test all baseline models')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force re-run even if test_results.csv exists')
    args = parser.parse_args()

    if args.force:
        args.skip_existing = False

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.backends.cudnn.enabled = False

    tests = [
        ('CPI_prediction', test_cpi_prediction),
        ('TransformerCPI', test_transformer_cpi),
        ('MPNN_CNN', test_mpnn_cnn),
        ('MSFF_DTA', test_msff_dta),
        ('AttentionSiteDTI', test_attentionsitedti),
    ]

    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        try:
            metrics = fn(device)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All baseline tests complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
