# AttentionSiteDTI

Attention-based Site DTI prediction model using TAGConv and BiLSTM with multi-head attention.

## Architecture
- Drug encoder: TAGConv (Topology-Adaptive Graph Convolution, 5 layers)
- Protein encoder: TAGConv (5 layers) + BiLSTM
- Interaction: Multi-head attention over concatenated drug-protein sequence
- Prediction head: MLP (8680 -> 4340 -> 1)

## Reference
Adapted from AttentionSiteDTI (InterpretableDTIP baseline).

## Requirements
- PyTorch
- torch_geometric (PyG)
- scikit-learn
- numpy
- tqdm

## Data
This model uses the same PyG graph data as the main 3D_GLAD model. 
Ensure `../3D_GLAD/cache/attention_data.pkl` exists (run the 3D_GLAD data preprocessing first).

## Training
```bash
python train.py
```

## Results (Best Epoch: 10)

| Split | AUC | AUPRC | F1 | Accuracy |
|-------|-----|-------|----|----------|
| Dev | 0.8865 | 0.8640 | 0.8327 | 0.8252 |
| Test | 0.8862 | 0.8526 | 0.8367 | 0.8315 |
