# Decoder Ablation Study

Comparing different decoder heads with GCN encoder and GAE pre-training.

All models use:
- GCN encoder (5 layers)
- GAE pre-training (30 epochs)
- Encoder freezing for first 3 epochs
- Early stopping patience: 15

## Decoder Descriptions
- **MLP**: Multi-layer perceptron with [concat, hadamard, diff] features
- **Bilinear**: Low-rank bilinear interaction (rank=32)
- **NTN**: Neural Tensor Network with 16 tensor slices
- **NodeBilinear**: Node-level bilinear interaction with attention pooling

## Results

| Decoder | Dev AUC | Best Epoch |
|---------|---------|------------|
| NTN | 0.9337 | 33 |
| MLP | 0.9307 | 20 |
| NodeBilinear | 0.9266 | 17 |
| Bilinear | 0.9202 | 58 |

Note: CrossAttn is our final decoder choice (included in the main model, not here).

## Training
```bash
python train.py --decoder mlp
python train.py --decoder bilinear
python train.py --decoder ntn
python train.py --decoder nodebilinear
```

## Data
Uses the same graph data as the main model. Ensure `../3D_GLAD/cache/attention_data.pkl` exists.
