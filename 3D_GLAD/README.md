# 3DGLAD: GCN + CrossAttn + GAE for Drug-Target Interaction Prediction

## Model Architecture

- **Drug Encoder**: 5-layer GCN (74 -> 70 -> 65 -> 60 -> 55 -> 31)
- **Protein Encoder**: 5-layer GCN (31 -> 31 x 5)
- **Decoder**: Per-Pocket Cross-Attention with gated pooling
- **Pre-training**: Graph Autoencoder (GAE) on both drug and protein graphs


## File Structure

```
3DGLAD/
├── model.py              # DTIModel: GCN encoder + CrossAttn decoder + GAE
├── data_preprocess.py    # Data loading, graph construction, pocket detection
├── train.py              # Full 2-phase training (GAE pre-train + DTI fine-tune)
├── infer.py              # Inference: CSV input -> prediction scores
├── best_model.pt         # Best checkpoint
├── uniprot_to_pdb.txt    # UniProt ID to PDB ID mapping
├── environment.yml       # Conda environment spec
├── requirements.txt      # Pip dependencies with versions
└── README.md
```

## Environment Setup

```bash
# Option 1: From conda environment file
conda env create -f environment.yml
conda activate deepdti

# Option 2: Create manually
conda create -n deepdti python=3.10 -y
conda activate deepdti
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.9.1+cu128 |
| torch_geometric | 2.7.0 |
| torch_scatter | 2.1.2+pt29cu128 |
| torch_sparse | 0.6.18+pt29cu128 |
| torch_cluster | 1.6.3+pt29cu128 |
| rdkit | 2026.03.1 |
| deepchem | 2.8.0 |
| biopython | 1.87 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| tqdm | 4.67.3 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |

## Training

### Phase 1: GAE Pre-training
Graph autoencoders pre-train drug and protein encoders via link prediction reconstruction.

### Phase 2: DTI Fine-tuning
- First 3 epochs: freeze encoders, train decoder only (lr=1e-3)
- After epoch 3: unfreeze all, joint training (lr=5e-4)
- Gradient accumulation: 16 steps
- Early stopping: patience=15 on Dev AUC
- Optimizer: Adam, gradient clipping=1.0

```bash
# From scratch
python train.py --data_dir /path/to/InterpretableDTIP/data --gae_epochs 30

# Resume training
python train.py --data_dir /path/to/InterpretableDTIP/data --resume logs/resume_model.pt
```

## Inference

The inference script reads a CSV file where:
- **Rows** = drug SMILES strings (row index)
- **Columns** = PDB IDs
- **Cells** = predicted binding scores

Proteins are encoded once and cached, then combined with each drug for efficient inference.
Missing PDB files are automatically downloaded from RCSB.

### Input CSV Format Example

Save as `input.csv`:

```csv
,7QGI,7R2V,7QIF
COC1=C(F)C=C(C=C1)C(=O)NC1CCN(CC1)C1=CC(=NN1)C1=CC=C(Cl)C=C1,,,
CC(C)C(=O)NC1CCC(CC1)NC1=NC(=NC2=C1C=NN2C1=CC=CC=C1)C1CC1,,,
CCC1=NN=C2N1C(NC1=CC=C(F)C=C1F)=NC1=CC=CC=C21,,,
```

### Run Inference

```bash
python infer.py --input input.csv --output predictions.csv
```

### Output

`predictions.csv` with the same shape as input, cells filled with binding probability scores [0, 1].

### PDB Files

Protein structures are stored as `{PDB_ID}.pdb` files in `cache/pdbs/` (or specify with `--pdb_dir`). Missing PDB files are automatically downloaded from RCSB. The model uses deepchem's ConvexHullPocketFinder to detect binding pockets and constructs per-pocket protein graphs.

### Training Dataset

BindingDB from https://github.com/IBM/InterpretableDTIP 

