import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader, SequentialSampler

DEEPPURPOSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Compare_models', 'DeepPurpose')
sys.path.insert(0, DEEPPURPOSE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV with SMILES,Target columns')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV with SMILES,Target,Prediction columns')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g. cuda, cpu, cuda:0)')
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from DeepPurpose import utils
    from DeepPurpose import DTI as models
    from DeepPurpose.utils import data_process_loader, mpnn_collate_func

    drug_encoding = 'MPNN'
    target_encoding = 'CNN'

    df_input = pd.read_csv(args.input)
    smiles = df_input['SMILES'].values
    targets = df_input['Target'].values
    dummy_labels = np.array([-1] * len(smiles))

    df_data = utils.data_process(
        smiles, targets, dummy_labels,
        drug_encoding, target_encoding,
        split_method='repurposing_VS', random_seed=42)

    result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'dp_results')
    os.makedirs(result_folder, exist_ok=True)

    config = utils.generate_config(
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        result_folder=result_folder,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100,
        LR=0.001,
        batch_size=256,
        hidden_dim_drug=128,
        mpnn_hidden_size=128,
        mpnn_depth=3,
        cnn_target_filters=[32, 64, 64],
        cnn_target_kernels=[4, 8, 8],
        num_workers=0,
        cuda_id=None,
    )
    config['decay'] = 0

    model = models.model_initialize(**config)
    model.device = device
    model.model = model.model.to(device)
    model.binary = True

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.ckpt}")

    info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **config)
    params = {'batch_size': config['batch_size'],
              'shuffle': False,
              'num_workers': 0,
              'drop_last': False,
              'collate_fn': mpnn_collate_func,
              'sampler': SequentialSampler(info)}
    generator = DataLoader(info, **params)

    model.model.eval()
    all_preds = []
    with torch.no_grad():
        for v_d, v_p, _ in generator:
            v_p = v_p.float().to(device)
            score = model.model(v_d, v_p)
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            all_preds.extend(logits.flatten().tolist())

    df_output = pd.DataFrame({
        'SMILES': smiles,
        'Target': targets,
        'Prediction': all_preds
    })
    df_output.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output} ({len(all_preds)} samples)")


if __name__ == '__main__':
    main()
