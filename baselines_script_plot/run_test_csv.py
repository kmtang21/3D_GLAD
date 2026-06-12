"""
Generate test_results.csv (Label, Prediction) for each baseline model.
Runs inference on the test set and saves predictions alongside labels.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled = False


def save_csv(y_true, y_pred, path):
    df = pd.DataFrame({'Label': np.array(y_true), 'Prediction': np.array(y_pred)})
    df.to_csv(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)")


# ==============================================================
# TransformerCPI
# ==============================================================
def run_transformer_cpi():
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'TransformerCPI')
    csv_path = os.path.join(model_dir, 'test_results.csv')
    if os.path.exists(csv_path):
        print("  [TransformerCPI] already exists, skip"); return

    sys.path.insert(0, model_dir)
    from model import (Encoder, Decoder, Predictor, pack,
                        DecoderLayer, SelfAttention, PositionwiseFeedforward)

    device = torch.device(DEVICE)
    test_pkl = os.path.join(model_dir, 'pre_data', 'test.pkl')
    if not os.path.exists(test_pkl):
        test_pkl = os.path.join(model_dir, 'data', 'test.pkl')
    with open(test_pkl, 'rb') as f:
        dataset = pickle.load(f)
    print(f"  Test: {len(dataset)}")

    encoder = Encoder(100, 64, 3, 9, 0.1, device)
    decoder = Decoder(34, 64, 3, 8, 256, DecoderLayer, SelfAttention, PositionwiseFeedforward, 0.1, device)
    model = Predictor(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device, weights_only=False))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="TransformerCPI"):
            atom, adj, protein, label = dataset[i]
            data_pack = pack([atom], [adj], [protein], [label], device)
            _, _, scores = model(data_pack, train=False)
            y_pred.append(scores[0])
            y_true.append(label if isinstance(label, int) else label.item())

    save_csv(y_true, y_pred, csv_path)
    del model; torch.cuda.empty_cache()


# ==============================================================
# CPI_prediction
# ==============================================================
def run_cpi_prediction():
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'CPI_prediction')
    csv_path = os.path.join(model_dir, 'test_results.csv')
    if os.path.exists(csv_path):
        print("  [CPI_prediction] already exists, skip"); return

    device = torch.device(DEVICE)
    test_pkl = None
    for d in [os.path.join(model_dir, 'pre_data'), os.path.join(model_dir, 'data')]:
        p = os.path.join(d, 'test.pkl')
        if os.path.exists(p):
            test_pkl = p; break
    if test_pkl is None:
        print("  No test.pkl, loading raw data...")
        sys.path.insert(0, os.path.join(PROJECT_DIR, 'baselines'))
        from data_loader import (load_all_data, build_unified_maps,
                                  smiles_to_atom_features_adj,
                                  protein_seq_to_onehot, AtomFeatureDataset)
        data_dir = os.path.join(PROJECT_DIR, 'InterpretableDTIP', 'data')
        train_data, dev_data, test_data = load_all_data(data_dir)
        us, useqs, uc2i, up2i = build_unified_maps(train_data, dev_data, test_data)
        max_atoms, max_prot_len = 100, 1000
        df = np.zeros((len(us), max_atoms, 26), dtype=np.float32)
        da = np.zeros((len(us), max_atoms, max_atoms), dtype=np.float32)
        dn = np.zeros(len(us), dtype=np.int64)
        for i, smi in enumerate(tqdm(us, desc="Drug feat")):
            f, a, n = smiles_to_atom_features_adj(smi, max_atoms)
            if f is not None: df[i]=f; da[i]=a; dn[i]=n
        po = np.zeros((len(useqs), max_prot_len, 22), dtype=np.float32)
        pl = np.zeros(len(useqs), dtype=np.int64)
        for i, seq in enumerate(tqdm(useqs, desc="Prot feat")):
            o, sl = protein_seq_to_onehot(seq, max_prot_len)
            po[i]=o; pl[i]=sl
        dataset = AtomFeatureDataset(df, da, dn, po, pl, uc2i, up2i, test_data[4], test_data[5])
    else:
        with open(test_pkl, 'rb') as f:
            dataset = pickle.load(f)
    print(f"  Test: {len(dataset)}")

    import torch.nn as nn
    class CPIModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_atom = nn.Linear(26, 64)
            self.embed_protein = nn.Linear(22, 64)
            self.W_gnn = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
            self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, 7, padding=3) for _ in range(3)])
            self.W_attention = nn.Linear(64, 64)
            self.W_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(3)])
            self.W_interaction = nn.Linear(128, 2)

        def gnn(self, xs, adj, layer):
            for i in range(layer):
                hs = torch.relu(self.W_gnn[i](xs))
                xs = xs + torch.matmul(adj, hs)
            return torch.unsqueeze(torch.mean(xs, dim=0), 0)

        def attention_cnn(self, x, xs, layer):
            xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
            for i in range(layer):
                xs = torch.relu(self.W_cnn[i](xs))
            xs = torch.squeeze(torch.squeeze(xs, 0), 0)
            h = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(xs))
            weights = torch.tanh(F.linear(h, hs))
            ys = torch.t(weights) * hs
            return torch.unsqueeze(torch.mean(ys, dim=0), 0)

        def forward(self, atom_feat, adj, atom_num, protein_oh, protein_len):
            atom_emb = torch.relu(self.embed_atom(atom_feat[:atom_num]))
            compound_vec = self.gnn(atom_emb, adj[:atom_num, :atom_num], 3)
            protein_emb = torch.relu(self.embed_protein(protein_oh[:protein_len]))
            protein_vec = self.attention_cnn(compound_vec, protein_emb, 3)
            cat_vec = torch.cat((compound_vec, protein_vec), 1)
            for j in range(3):
                cat_vec = torch.relu(self.W_out[j](cat_vec))
            return self.W_interaction(cat_vec)

    model = CPIModel().to(device)
    ckpt = torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for feat, adj, atom_num, protein_oh, protein_len, label in tqdm(loader, desc="CPI_prediction"):
            feat, adj, protein_oh = feat.to(device), adj.to(device), protein_oh.to(device)
            for i in range(feat.size(0)):
                out = model(feat[i], adj[i], atom_num[i], protein_oh[i], protein_len[i])
                prob = F.softmax(out, dim=1)[0, 1].item()
                y_pred.append(prob)
                y_true.append(label[i].item())

    save_csv(y_true, y_pred, csv_path)
    del model; torch.cuda.empty_cache()


# ==============================================================
# MPNN_CNN
# ==============================================================
def run_mpnn_cnn():
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'MPNN_CNN')
    csv_path = os.path.join(model_dir, 'test_results.csv')
    if os.path.exists(csv_path):
        print("  [MPNN_CNN] already exists, skip"); return

    device = torch.device(DEVICE)
    DP_DIR = os.path.join(PROJECT_DIR, 'Compare_models', 'DeepPurpose')
    sys.path.insert(0, DP_DIR)
    sys.path.insert(0, model_dir)

    from DeepPurpose import utils
    from DeepPurpose import DTI as models
    from DeepPurpose.utils import data_process_loader, mpnn_collate_func
    from torch.utils.data import DataLoader, SequentialSampler

    def load_csv(filepath):
        df = pd.read_csv(filepath)
        return df['SMILES'].values, df['Target'].values, df['Label'].values

    pre_data_dir = os.path.join(model_dir, 'pre_data')
    data_subdir = os.path.join(model_dir, 'data')
    DATA_DIR = pre_data_dir if os.path.exists(os.path.join(pre_data_dir, 'test.csv')) else data_subdir

    test_df = utils.data_process(*load_csv(os.path.join(DATA_DIR, 'test.csv')),
                                  'MPNN', 'CNN', split_method='no_split', random_seed=42)
    print(f"  Test: {len(test_df)}")

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
    ckpt = torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt['model_state_dict'])

    test_info = data_process_loader(test_df.index.values, test_df.Label.values, test_df, **config)
    loader = DataLoader(test_info, batch_size=config['batch_size'], shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=mpnn_collate_func,
                        sampler=SequentialSampler(test_info))

    model.model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for v_d, v_p, label in tqdm(loader, desc="MPNN_CNN"):
            v_p = v_p.float().to(device)
            score = model.model(v_d, v_p)
            logits = torch.squeeze(torch.sigmoid(score)).detach().cpu().numpy()
            y_pred.extend(logits.flatten().tolist())
            y_true.extend(np.array(label).flatten().tolist())

    save_csv(y_true, y_pred, csv_path)
    del model; torch.cuda.empty_cache()


# ==============================================================
# AttentionSiteDTI
# ==============================================================
def run_attentionsitedti():
    model_dir = os.path.join(PROJECT_DIR, 'Baselines', 'AttentionSiteDTI')
    csv_path = os.path.join(model_dir, 'test_results.csv')
    if os.path.exists(csv_path):
        print("  [AttentionSiteDTI] already exists, skip"); return

    device = torch.device(DEVICE)
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
            self.bilstm = torch.nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=0.0)
            self.fc_in = torch.nn.Linear(8680, 4340)
            self.fc_out = torch.nn.Linear(4340, 1)
            self.attention = MultiHeadAttention(62, 62, 2)

        def forward(self, pg, dg, device):
            fp = pg.x; fd = dg.x; pe = pg.edge_index; de = dg.edge_index
            for m in self.protein_graph_conv: fp = F.relu(m(fp, pe))
            for m in self.ligand_graph_conv: fd = F.relu(m(fd, de))
            pr = self.pooling_protein(fp, pg.batch).view(-1, 31)
            db = torch.zeros(fd.size(0), dtype=torch.long, device=device)
            lr = self.pooling_ligand(fd, db).view(-1, 31)
            seq = torch.cat((lr, pr), dim=0).view(1, -1, 31)
            sl = seq.size(1)
            mask = torch.eye(140, dtype=torch.bool).view(1, 140, 140).to(device)
            mask[0, sl:140, :] = False; mask[0, :, sl:140] = False
            mask[0, :, sl-1] = True; mask[0, sl-1, :] = True; mask[0, sl-1, sl-1] = False
            seq = F.pad(seq, (0, 0, 0, 140-sl), value=0).permute(1, 0, 2)
            h0 = torch.zeros(2, 1, 31).to(device); c0 = torch.zeros(2, 1, 31).to(device)
            out, _ = self.bilstm(seq, (h0, c0)); out = out.permute(1, 0, 2)
            out = self.attention(out, mask=mask)
            out = F.relu(self.fc_in(out.view(-1, out.size(1)*out.size(2))))
            return torch.sigmoid(self.fc_out(out))

    cache = os.path.join(model_dir, 'pre_data', 'attention_data.pkl')
    if not os.path.exists(cache):
        cache = os.path.join(PROJECT_DIR, '3D_GLAD', 'cache', 'attention_data.pkl')
    sys.path.insert(0, os.path.join(PROJECT_DIR, '3D_GLAD'))
    with open(cache, 'rb') as f:
        data = pickle.load(f)
    test_ds = data['test']
    print(f"  Test: {len(test_ds)}")

    ckpt_path = os.path.join(PROJECT_DIR, 'AttentionSiteDTI', 'interpretable_dtip_output', 'best_model.pt')
    model = DTITAG().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded ckpt from {ckpt_path}")

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="AttentionSiteDTI"):
            pg, dg, label = test_ds[i]
            pg, dg = pg.to(device), dg.to(device)
            out = model(pg, dg, device)
            y_pred.append(out.cpu().item())
            y_true.append(label)

    save_csv(y_true, y_pred, csv_path)
    del model; torch.cuda.empty_cache()


# ==============================================================
def main():
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument('--models', nargs='*', default=None,
                   help='Specific models to run (TransformerCPI CPI_prediction MPNN_CNN AttentionSiteDTI)')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    runners = {
        'TransformerCPI': run_transformer_cpi,
        'CPI_prediction': run_cpi_prediction,
        'MPNN_CNN': run_mpnn_cnn,
        'AttentionSiteDTI': run_attentionsitedti,
    }

    if args.force:
        for name in runners:
            csv = os.path.join(PROJECT_DIR, 'Baselines', name, 'test_results.csv')
            if os.path.exists(csv):
                os.remove(csv)

    to_run = args.models if args.models else list(runners.keys())

    for name in to_run:
        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        try:
            runners[name]()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == '__main__':
    main()
