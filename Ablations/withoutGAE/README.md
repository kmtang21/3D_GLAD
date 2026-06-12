# GAE Ablation Study

Comparing GCN+CrossAttn with and without GAE pre-training.

## Files
- `best_GCN_crossattn_noGAE.pt` — GCN+CrossAttn **without** GAE pre-training (Dev AUC: 0.9242)

## Configuration
- Encoder: GCN (5 layers)
- Decoder: CrossAttn
- GAE pre-training: 30 epochs (drug + protein)
- Encoder freezing for first 3 epochs
- Early stopping patience: 15

## Results

| Configuration | GAE | Dev AUC | Best Epoch |
|---------------|-----|---------|------------|
| GCN+CrossAttn | No | 0.9242 | 7 |

GAE pre-training improves Dev AUC by +0.0188.

## Training (no-GAE variant)
```bash
python train.py --conv GCN
```

## Data
Uses the same graph data as the main model. Ensure `../3D_GLAD/cache/attention_data.pkl` exists.
