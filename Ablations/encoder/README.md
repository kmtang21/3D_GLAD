# Encoder Ablation Study

Comparing different GNN encoders with GAE pre-training and CrossAttn decoder.

All models use:
- GAE pre-training (30 epochs)
- CrossAttn decoder
- Encoder freezing for first 3 epochs
- Early stopping patience: 15

## Results

| Encoder | Dev AUC | Best Epoch | Parameters |
|---------|---------|------------|------------|
| TAG | 0.9270 | 31 | - |
| Transformer | 0.9260 | 25 | - |
| GCN | 0.9259 | 14 | - |
| GIN | 0.9226 | 7 | - |
| GATv2 | 0.8453 | 7 | - |

Note: GCN is our final encoder choice (included in the main model, not here).

## Training
```bash
# Train with specific encoder
python train.py --conv TAG
python train.py --conv GATv2
python train.py --conv GIN
python train.py --conv Transformer
```

## Data
Uses the same graph data as the main model. Ensure `../3D_GLAD/cache/attention_data.pkl` exists.
