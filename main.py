'''
=== Best Hyperparameters ===
Model: UNET256
Params: {'batch_size': 4, 'epochs': 5, 'filter_size': 5, 'learning_rate': 0.001, 'padding': 1, 'stride': 3}
Val Loss: 0.0519

Model: UNET512
Params: {'batch_size': 4, 'epochs': 5, 'filter_size': 4, 'learning_rate': 0.001, 'padding': 2, 'stride': 3}
Val Loss: 0.0380

=== Final Training ===
time uv run python3 main.py --verbose train --model UNET512 --epochs 5 --filter-size 4 --lr 0.001 --stride 3 --padding 2

[Epoch 01] Train Loss: 0.2254
           Val Loss:   0.0456
Mean Absolute Error:  [0.00419942 0.00259909 0.00022896]
Tissue composition for a single image:
  Predicted:     Adipose: 39.35%, Fibro: 22.46%, Calc: 0.06%
  Ground Truth:  Adipose: 39.67%, Fibro: 22.35%, Calc: 0.04%
---------------------
[Epoch 02] Train Loss: 0.0478
           Val Loss:   0.0366
Mean Absolute Error:  [4.6931258e-03 4.6866811e-03 9.8491204e-05]
Tissue composition for a single image:
  Predicted:     Adipose: 40.12%, Fibro: 21.89%, Calc: 0.03%
  Ground Truth:  Adipose: 39.67%, Fibro: 22.35%, Calc: 0.04%
---------------------
[Epoch 03] Train Loss: 0.0365
           Val Loss:   0.0331
Mean Absolute Error:  [1.6103401e-03 1.7545491e-03 4.7579295e-05]
Tissue composition for a single image:
  Predicted:     Adipose: 39.51%, Fibro: 22.52%, Calc: 0.05%
  Ground Truth:  Adipose: 39.67%, Fibro: 22.35%, Calc: 0.04%
---------------------
[Epoch 04] Train Loss: 0.0324
           Val Loss:   0.0310
Mean Absolute Error:  [2.1582521e-03 2.5966563e-03 4.1078700e-05]
Tissue composition for a single image:
  Predicted:     Adipose: 39.83%, Fibro: 22.14%, Calc: 0.05%
  Ground Truth:  Adipose: 39.67%, Fibro: 22.35%, Calc: 0.04%
---------------------
[Epoch 05] Train Loss: 0.0304
           Val Loss:   0.0292
Mean Absolute Error:  [2.6003474e-03 2.7826561e-03 3.4554600e-05]
Tissue composition for a single image:
  Predicted:     Adipose: 39.42%, Fibro: 22.61%, Calc: 0.04%
  Ground Truth:  Adipose: 39.67%, Fibro: 22.35%, Calc: 0.04%
---------------------

real	9m32.029s
user	9m38.294s
sys	0m20.950s

'''


import sys
import argparse
import torch.nn as nn
from sklearn.model_selection import ParameterGrid

# Import your new Dataset and Trainer
from src.EDA          import run_eda
from src.Trainer      import Trainer
from src.CNN          import UNet256, UNet512
from src.AttnDataset  import AttnDataset

def main():
    parser = argparse.ArgumentParser(
        prog='spectral_ct',
        description='Run EDA or training on the PINN.'
    )
    args = do_options_parsing(parser)

    # Load dataset
    ds = AttnDataset(args.datadir)

    # Mode dispatch
    if args.mode == 'eda':
        run_eda(args, ds.train_ds)

    elif args.mode == 'tune':
        # Define search spaces for each model
        search_spaces = {
            'UNET256': {
                'filter_size': [3, 4, 5],
                'stride':      [2, 3],
                'padding':     [1, 2],
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'batch_size':    [4, 8],
                'epochs':        [3, 5]
            },
            'UNET512': {
                'filter_size': [3, 4, 5],
                'stride':      [2, 3],
                'padding':     [1, 2],
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'batch_size':    [4, 8],
                'epochs':        [3, 5]
            }
        }

        train_ds = ds.subsample(args.subsample)

        best_overall = {'loss': float('inf'), 'params': None, 'model': None}

        # Choose the grid for the requested model
        grid = search_spaces.get(args.model.upper())
        if grid is None:
            print(f"Unknown model for tuning: {args.model}")
            sys.exit(1)

        # Iterate combinations
        for params in ParameterGrid(grid):
            print(f"Evaluating {args.model} with {params}")
            # Instantiate model dynamically
            model_cls = UNet256 if args.model.upper() == 'UNET256' else UNet512
            model = model_cls(
                in_channels=2,
                out_channels=3,
                kernel_size=params['filter_size'],
                stride=params['stride'],
                padding=params['padding']
            )

            trainer = Trainer(
                model=model,
                train_dataset=train_ds,
                val_dataset=ds.test_ds,
                batch_size=params['batch_size'],
                lr=params['learning_rate'],
                loss_fn=nn.BCELoss(),
                num_workers=4
            )
            # Warm-up
            for _ in range(params['epochs']):
                trainer.train_epoch()
            val_loss = trainer.evaluate()

            if val_loss < best_overall['loss']:
                best_overall.update({'loss': val_loss, 'params': params.copy(), 'model': args.model})

        print("=== Best Hyperparameters ===")
        print(f"Model: {best_overall['model']}")
        print(f"Params: {best_overall['params']}")
        print(f"Val Loss: {best_overall['loss']:.4f}")
        return

    elif args.mode == 'train':
        if args.model == 'UNET256':
            model = UNet256(in_channels=2, out_channels=3)
            trainer = Trainer(
                model=model,
                train_dataset=ds.train_ds,
                val_dataset=ds.test_ds,
                batch_size=args.batch_size,
                lr=args.lr,
                loss_fn=None,
                device=None,
                num_workers=4,
                verbose=args.verbose
            )
        elif args.model == 'UNET512':
            model = UNet512(in_channels=2, out_channels=3)
            trainer = Trainer(
                model=model,
                train_dataset=ds.train_ds,
                val_dataset=ds.test_ds,
                batch_size=args.batch_size,
                lr=args.lr,
                loss_fn=None,
                device=None,
                num_workers=4,
                verbose=args.verbose
            )

        else:
            print('Invalid model: choose UNET256 or UNET512.')
            sys.exit(1)

        trainer.run(epochs=args.epochs)

    elif args.mode == 'skip':
        print('Skipping all modes.')


def do_options_parsing(parser):

    parser.add_argument('--datadir', type=str, default='./data', help='Path to data directory.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    subparsers = parser.add_subparsers(dest='mode', required=True)

    # EDA
    p_eda = subparsers.add_parser('eda', help='Exploratory data analysis')
    p_eda.add_argument('--nsamples', type=int, default=100000,
                       help='Number of pixel samples to plot')

    # Train
    p_train = subparsers.add_parser('train', help='Train the model')
    p_train.add_argument('--model',       type=str,   default='UNET256',
                         help='Model type: UNET256, UNET512')
    p_train.add_argument('--epochs',      type=int,   default=5,
                         help='Training epochs')
    p_train.add_argument('--filter-size', type=int,   default=3,
                         help='CNN filter size.')
    p_train.add_argument('--batch-size',  type=int,   default=8,
                         help='Batch size')
    p_train.add_argument('--lr',          type=float, default=1e-3,
                         help='Learning rate')
    p_train.add_argument('--stride',      type=float, default=2,
                         help='Physics-loss weight for MLPs')
    p_train.add_argument('--padding',     type=float, default=1,
                         help='Physics-loss weight for MLPs')
    
    # Hyperparameter tuning
    p_tune = subparsers.add_parser('tune', help='Hyperparameter tuning')
    p_tune.add_argument('--subsample',  type=float, default=0.1,
                         help='Fraction of training set to use')
    p_tune.add_argument('--model',       type=str,   default='UNET256',
                         help='Model type: UNET256, UNET512')

    # Skip
    subparsers.add_parser('skip', help='Skip execution')

    return parser.parse_args()

if __name__ == "__main__":
    main()
