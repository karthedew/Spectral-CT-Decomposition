import os
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.Dataset import TissueSegmentationDataset
from src.CNN import BasicUNet
from src.Trainer import Trainer

# Directory containing your data
DATA_DIR = './data'

# Number of epochs for each trial (keep small for tuning)
N_EPOCHS = 5

# Use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    # Prepare datasets
    train_ds = TissueSegmentationDataset(DATA_DIR, split='train')
    val_ds   = TissueSegmentationDataset(DATA_DIR, split='val')

    # Instantiate model and trainer
    model = BasicUNet(in_channels=2, out_channels=3).to(DEVICE)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        lr=lr,
        loss_fn=nn.BCELoss(),
        device=DEVICE,
        num_workers=4
    )

    # Warm-start training for a few epochs
    for epoch in range(1, N_EPOCHS + 1):
        trainer.train_epoch()
    # Evaluate on validation
    val_loss = trainer.evaluate()

    return val_loss


def tune(n_trials: int = 20):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value:.4f}')
    print('  Params:')
    for key, val in trial.params.items():
        print(f'    {key}: {val}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--trials', type=int, default=20, help='Number of tuning trials')
    args = parser.parse_args()
    tune(n_trials=args.trials)
