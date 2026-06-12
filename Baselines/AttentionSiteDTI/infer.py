import os
import sys
import pickle
import csv
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, GlobalAttention
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from layers import MultiHeadAttention

PYG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '3D_GLAD', 'cache', 'attention_data.pkl')
LOCAL_CACHE = os.path.join(os.path.dirname(__file__), 'pre_data', 'attention_data.pkl')
PYG_CACHE = LOCAL_CACHE if os.path.exists(LOCAL_CACHE) else PYG_CACHE
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'logs')
INTERPRETABLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'InterpretableDTIP', 'data')


class DTITAG(nn.Module):
    def __init__(self):
        super(DTITAG, self).__init__()
        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, K=2))

        self.ligand_graph_conv = nn.ModuleList()
        self.ligand_graph_conv.append(TAGConv(74, 70, K=2))
        self.ligand_graph_conv.append(TAGConv(70, 65, K=2))
        self.ligand_graph_conv.append(TAGConv(65, 60, K=2))
        self.ligand_graph_conv.append(TAGConv(60, 55, K=2))
        self.ligand_graph_conv.append(TAGConv(55, 31, K=2))

        self.pooling_protein = GlobalAttention(nn.Linear(31, 1))
        self.pooling_ligand = GlobalAttention(nn.Linear(31, 1))
        self.dropout_rate = 0.2
        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=0.0)
        self.fc_in = nn.Linear(8680, 4340)
        self.fc_out = nn.Linear(4340, 1)
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


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def load_raw_split(split_name):
    data_dir = os.path.join(INTERPRETABLE_DATA_DIR, split_name)

    chem_ids = []
    with open(os.path.join(data_dir, 'chem'), 'r') as f:
        for line in f:
            chem_ids.append(line.strip())

    chem_smiles = []
    with open(os.path.join(data_dir, 'chem.repr'), 'r') as f:
        for line in f:
            chem_smiles.append(line.strip())

    protein_ids = []
    with open(os.path.join(data_dir, 'protein'), 'r') as f:
        for line in f:
            protein_ids.append(line.strip())

    protein_repr = []
    with open(os.path.join(data_dir, 'protein.repr'), 'r') as f:
        for line in f:
            protein_repr.append([int(x) for x in line.strip().split()])

    chem_id_to_smiles = {cid: chem_smiles[i] for i, cid in enumerate(chem_ids)}
    protein_id_to_seq = {}
    for i, pid in enumerate(protein_ids):
        seq_str = ''.join(AMINO_ACIDS[idx] if idx < len(AMINO_ACIDS) else 'X' for idx in protein_repr[i])
        protein_id_to_seq[pid] = seq_str

    edges = []
    labels = []
    with open(os.path.join(data_dir, 'edges.pos'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(1)
    with open(os.path.join(data_dir, 'edges.neg'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(0)

    return chem_id_to_smiles, protein_id_to_seq, edges, labels


def main():
    parser = argparse.ArgumentParser(description='AttentionSiteDTI Inference')
    parser.add_argument('--ckpt', type=str, default=os.path.join(OUTPUT_DIR, 'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g. cuda:0, cpu)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'dev'],
                        help='Which data split to run inference on')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV with SMILES,Target columns (matched against cached data)')
    parser.add_argument('--output', type=str, default=os.path.join(OUTPUT_DIR, 'inference_results.csv'),
                        help='Output CSV path')
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DTITAG().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}")

    print(f"Loading data from {PYG_CACHE}")
    with open(PYG_CACHE, 'rb') as f:
        data = pickle.load(f)

    split = args.split
    dataset = data[split]
    print(f"Dataset ({split}) size: {len(dataset)}")

    chem_id_to_smiles, protein_id_to_seq, edges, labels_list = load_raw_split(split)

    if len(dataset) != len(edges):
        print(f"WARNING: dataset size ({len(dataset)}) != edges size ({len(edges)})")
        n_samples = min(len(dataset), len(edges))
    else:
        n_samples = len(dataset)

    input_pairs = None
    if args.input is not None:
        input_pairs = set()
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row['SMILES'].strip()
                seq = row['Target'].strip()
                input_pairs.add((smi, seq))
        print(f"Loaded {len(input_pairs)} input pairs for matching")

    results = []
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Inference", leave=False):
            chem_id, protein_id = edges[i]
            label = labels_list[i]

            smiles = chem_id_to_smiles.get(chem_id, '')
            seq = protein_id_to_seq.get(protein_id, '')

            if input_pairs is not None:
                if (smiles, seq) not in input_pairs:
                    continue

            pgraph, dgraph, _ = dataset[i]
            pgraph = pgraph.to(device)
            dgraph = dgraph.to(device)
            out = model(pgraph, dgraph, device)
            pred = out.cpu().item()

            results.append({
                'SMILES': smiles,
                'Target': seq,
                'Prediction': pred,
                'Label': label,
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SMILES', 'Target', 'Prediction', 'Label'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nInference complete: {len(results)} samples")
    if len(results) > 0:
        preds = np.array([r['Prediction'] for r in results])
        labels_arr = np.array([r['Label'] for r in results])
        print(f"  Mean prediction: {preds.mean():.4f}")
        print(f"  Positive labels: {labels_arr.sum()}/{len(labels_arr)}")
    print(f"Results saved to {args.output}")

    if input_pairs is not None:
        print(f"  Matched {len(results)}/{len(input_pairs)} input pairs from cached data")
        if len(results) == 0 and len(input_pairs) > 0:
            print("  NOTE: No matches found. For new SMILES/protein pairs, "
                  "run the 3D_GLAD preprocessing pipeline first to generate PyG graphs.")


if __name__ == '__main__':
    main()
