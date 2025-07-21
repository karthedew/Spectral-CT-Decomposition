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
                num_workers=4
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
                num_workers=4
            )

        else:
            print('Invalid model: choose UNET256 or UNET512.')
            sys.exit(1)

        trainer.run(epochs=args.epochs)

    elif args.mode == 'skip':
        print('Skipping all modes.')


def do_options_parsing(parser):

    parser.add_argument('--datadir', type=str, default='./data', help='Path to data directory.')

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
