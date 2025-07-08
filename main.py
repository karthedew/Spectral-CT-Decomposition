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

def main():
    data_dir = './data'
    ds = MLPTestTrainDataset(data_dir)
    trainer = CosMLPTrainer(ds.train_ds, ds.test_ds, ds.prototypes)
    trainer.run(epochs=10)

def do_options_parsing(p):
    return p


if __name__ == "__main__":
    main()
