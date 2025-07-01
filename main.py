import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import astra

from src.Dataset import MLPTestTrainDataset
from src.MLP     import CosMLPTrainer

def main():
    data_dir = './data'
    ds = MLPTestTrainDataset(data_dir)
    trainer = CosMLPTrainer(ds.train_dataset, ds.test_dataset, ds.prototypes)
    trainer.run(epochs=10)


if __name__ == "__main__":
    main()
