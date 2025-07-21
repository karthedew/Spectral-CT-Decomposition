# PyTorch Dataset and DataLoader
import sys
import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from sklearn.model_selection import train_test_split


def load_npy_gz(path):
    '''
    Load a .npy.gz file and return the numpy array.
    '''
    with gzip.open(path, 'rb') as file:
        return np.load(file) #, mmap_mode='r')

class TissueSegmentationDataset(Dataset):
    def __init__(self, attn_vector, labels):
        self.x = attn_vector            # (1000, 2, 512, 512)
        self.y = labels                 # (1000, 3, 512, 512)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MuDataset(Dataset):
    def __init__(self, attn_vector, labels):
        self.x = attn_vector           # shape: (1000, 2, 256, 1024)
        self.y = labels                # shape: (1000, 3)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLPTestTrainDataset:
    '''
    Load spectral CT data, computes attenuation, and builds train/test datasets.

    Attributes:
        train_dataset (Dataset): training set
        test_dataset  (Dataset): testing set
    '''
    def __init__(self, data_dir: str, test_size: float = 0.2, random_state: int = 42, normalize: bool = False):

        # Identify file paths
        files = os.listdir(data_dir)
        high_tx_path = os.path.join(data_dir, 'highkVpTransmission.npy.gz')
        low_tx_path  = os.path.join(data_dir, 'lowkVpTransmission.npy.gz')
        adipose_path = os.path.join(data_dir, 'Phantom_Adipose.npy.gz')
        fibro_path   = os.path.join(data_dir, 'Phantom_Fibroglandular.npy.gz')
        calc_path    = os.path.join(data_dir, 'Phantom_Calcification.npy.gz')

        print('Reading Data Files...')
        # Load transmission sinograms
        high_tx = load_npy_gz(high_tx_path)  # (N, angles, pixels)
        low_tx  = load_npy_gz(low_tx_path)

        # Compute attenuation coefficients mu = -ln(I/I0), assume I0=1
        eps = 1e-6
        mu_high = -np.log(np.clip(high_tx, eps, None))
        mu_low  = -np.log(np.clip(low_tx,  eps, None))

        # Load binary masks for tissue labels
        adipose = load_npy_gz(adipose_path)  # (N, H, W)
        fibro   = load_npy_gz(fibro_path)
        calc    = load_npy_gz(calc_path)

        # Save the raw data for Obj Access
        self.high_tx = high_tx
        self.low_tx  = low_tx
        self.mu_high = mu_high
        self.mu_low  = mu_low
        self.attn    = np.stack([mu_low, mu_high], axis=1)
        self.adipose = adipose
        self.fibro   = fibro
        self.calc    = calc

        print('===============================')
        print('Mu High Shape: ', self.mu_high.shape)
        print('Mu Low Shape:  ', self.mu_low.shape)
        print('Atten Vector:  ', self.attn.shape)
        print('Adipose Shape: ', self.adipose.shape)
        print('Fibro Shape:   ', self.fibro.shape)
        print('Calc Shape:    ', self.calc.shape)
        print('===============================')

        labels = np.zeros((1000, 3), dtype=int)
        for i in range(0,self.adipose.shape[0]):
            has_adipose = np.any(adipose[i])
            has_fibro   = np.any(fibro[i])
            has_calc    = np.any(calc[i])

            # label -> adipose, fibro, calc
            if has_adipose: labels[i, 0] = 1
            if has_fibro:   labels[i, 1] = 1
            if has_calc:    labels[i, 2] = 1

        self.classification_labels = labels
        print('--------------------------')
        print('Check uniqueness of labels')
        print('Adipose: ', np.unique(self.adipose))
        print('Fibro:   ', np.unique(self.fibro))
        print('Calc:    ', np.unique(self.calc))
        print('Labels:  ', np.unique(self.classification_labels))
        print('--------------------------')
        if normalize:
            self.attn = self.attn / np.max(self.attn)

        self.labels = np.stack([
            self.adipose,
            self.fibro,
            self.calc
        ])
        self.tissue_segment_dataset = TissueSegmentationDataset(self.attn, self.labels)
        self.dataset = MuDataset(self.attn, self.labels)

        # Define split sizes
        train_size = int(0.8 * len(self.dataset))  # 800 samples
        test_size = len(self.dataset) - train_size # 200 samples

        # Split the dataset
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

        # Create DataLoaders
        self.train_ds = train_dataset
        self.test_ds  = test_dataset
        '''
        # Compute mean and std on training features
        global_mean = X_train.mean(axis=0)
        global_std  = X_train.std(axis=0) + 1e-6  # avoid zero division
        # Apply scaling
        self.X_train = (X_train - global_mean) / global_std
        self.X_test  = (X_test  - global_mean) / global_std

        prototypes = []
        for cls in sorted(np.unique(self.y_train)):
            mask = (self.y_train == cls)
            mu_low_mean  = self.X_train[mask, 0].mean()
            mu_high_mean = self.X_train[mask, 0].mean()
            prototypes.append([mu_low_mean, mu_high_mean])
        self.prototypes = prototypes # torch.tensor(prototypes) #.float().cuda()  # shape (C, 2)

        self.train_ds = TensorDataset(
            torch.from_numpy(self.X_train),
            torch.from_numpy(self.y_train)
        )
        self.test_ds  = TensorDataset(
            torch.from_numpy(self.X_test),
            torch.from_numpy(self.y_test)
        )
        '''

    def subsample(self, subsample_size: float = 0.01):
        full_size_train = len(self.train_ds)
        subset_size     = int(full_size_train * subsample_size)
        indices         = torch.randperm(full_size_train)[:subset_size]
        return Subset(self.train_ds, indices)


# Example usage:
# ds = MLPTestTrainDataset('./data')
# train_loader = DataLoader(ds.train_dataset, batch_size=4096, shuffle=True)
# test_loader  = DataLoader(ds.test_dataset,  batch_size=4096)
