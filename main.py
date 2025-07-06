import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import astra
import numpy as np

from src.Dataset import MLPTestTrainDataset
from src.PINN    import CosMLPTrainer

def main():
    data_dir = './data'
    ds = MLPTestTrainDataset(data_dir)
    trainer = CosMLPTrainer(ds.train_dataset, ds.test_dataset, ds.prototypes)
#    trainer.run(epochs=10)
    print(ds.high_tx.shape)


if __name__ == "__main__":
    main()
