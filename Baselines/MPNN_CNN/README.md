# MPNN_CNN (DeepPurpose)

Message Passing Neural Network + CNN model from DeepPurpose.

## Architecture
- Drug encoder: MPNN (Message Passing Neural Network)
- Protein encoder: CNN (1D Convolutional Neural Network)
- Prediction head: MLP classifier

## Reference
Huang K et al. "DeepPurpose: a Deep Learning Library for Drug-Target Interaction Prediction."

## Requirements
- PyTorch
- DeepPurpose (included in ../Compare_models/DeepPurpose/)
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

## Results (Best Epoch: 31)

| Split | AUC | AUPRC | F1 | Accuracy |
|-------|-----|-------|----|----------|
| Dev | 0.9148 | 0.9142 | 0.8348 | 0.8307 |
| Test | 0.9182 | 0.9116 | 0.8370 | 0.8326 |
