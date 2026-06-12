# CPI_prediction

GNN + Attention-CNN model for compound-protein interaction prediction.

## Architecture
- Drug encoder: GNN (Graph Neural Network with linear layers)
- Protein encoder: Attention-CNN (1D convolution + attention mechanism)
- Prediction head: MLP with concatenation of drug and protein representations

## Reference
Adapted from CPI_prediction baseline.

## Requirements
- PyTorch
- scikit-learn
- rdkit
- numpy
- tqdm

## Data Preprocessing
```bash
python data_preprocess.py --data_dir ../InterpretableDTIP/data --output_dir ./data
```

## Training
```bash
python train.py
```

## Results (Best Epoch: 65)

| Split | AUC | AUPRC | F1 | Accuracy |
|-------|-----|-------|----|----------|
| Dev | 0.9232 | 0.9223 | 0.8502 | 0.8460 |
| Test | 0.9231 | 0.9141 | 0.8458 | 0.8409 |
