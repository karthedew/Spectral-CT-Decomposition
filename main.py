import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import astra
import numpy as np

from src.Dataset import MLPTestTrainDataset
from src.PINN    import CosMLPTrainer
from src.EDA     import run_eda

def main():
    data_dir = './data'
    parser = argparse.ArgumentParser(
        prog='spectral_ct',
        description='Run EDA or training on the PINN.'
    )
    args = do_options_parsing(parser)
    ds = MLPTestTrainDataset(data_dir)

    '''
    mh_max = ds.mu_high.max()
    mh_min = ds.mu_high.min()
    ml_max = ds.mu_low.max()
    ml_min = ds.mu_low.min()

    print('MU HIGH MAX: ', mh_max)
    print('MU HIGH MIN: ', mh_min)
    print('MU LOW  MAX: ', ml_max)
    print('MU LOW  MIN: ', ml_min)
    '''

    if args.mode == 'eda':
        run_eda(args, ds)
    elif args.mode == 'train':
        train_ds_subsample = ds.subsample()
        trainer = CosMLPTrainer(
            train_dataset = train_ds_subsample,
            test_dataset  = ds.test_ds,
            prototypes    = ds.prototypes,
            batch_size    = args.batch_size,
            alpha         = args.alpha
        )
        trainer.run(epochs=args.epochs)


def do_options_parsing(parser):

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # EDA subcommand
    p_eda = subparsers.add_parser("eda", help="Generate exploratory data analysis plots")
    p_eda.add_argument("--nsamples", type=int, default=100000,
                       help="Number of pixel samples to plot")

    # Training subcommand
    p_train = subparsers.add_parser("train", help="Train the PINN model")
    p_train.add_argument("--batch-size", type=int, default=65539, help="Batch size")
    p_train.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p_train.add_argument("--alpha", type=float, default=0.5,
                         help="Physics-loss weight")

    return parser.parse_args()


if __name__ == "__main__":
    main()
