# TransformerCPI

Transformer-based CPI prediction model with GCN compound encoder and CNN protein encoder.

## Architecture
- Drug encoder: GCN (Graph Convolutional Network)
- Protein encoder: CNN (1D Convolution with GLU activation)
- Decoder: Transformer cross-attention decoder

## Reference
Chen L et al. "TransformerCPI: improving compound-protein interaction prediction by sequence-based deep learning with self-attention mechanism."

## Requirements
- PyTorch
- scikit-learn
- rdkit
- tqdm

## Data Preprocessing
```bash
python data_preprocess.py --data_dir ../InterpretableDTIP/data --output_dir ./data
```

## Training
```bash
python train.py
```

## Results (Best Epoch: 27)

| Split | AUC | AUPRC | F1 | Accuracy |
|-------|-----|-------|----|----------|
| Dev | 0.9357 | 0.9434 | 0.8690 | 0.8649 |
| Test | 0.9329 | 0.9361 | 0.8655 | 0.8616 |
