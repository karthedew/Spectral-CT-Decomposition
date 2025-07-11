# PyTorch Dataset and DataLoader
import sys
import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split


def load_npy_gz(path):
    '''
    Load a .npy.gz file and return the numpy array.
    '''
    with gzip.open(path, 'rb') as file:
        print(path)
        return np.load(file) #, mmap_mode='r')


class MLPTestTrainDataset:
    '''
    Load spectral CT data, computes attenuation, and builds train/test datasets.

    Attributes:
        train_dataset (Dataset): training set
        test_dataset  (Dataset): testing set
    '''
    def __init__(self, data_dir: str, test_size: float = 0.2, random_state: int = 42):

        # Identify file paths
        files = os.listdir(data_dir)
        high_tx_path = os.path.join(data_dir, 'highkVpTransmission.npy.gz')
        low_tx_path  = os.path.join(data_dir, 'lowkVpTransmission.npy.gz')
        adipose_path = next(os.path.join(data_dir, f) for f in files if 'Adipose' in f)
        fibro_path   = next(os.path.join(data_dir, f) for f in files if 'Fibroglandular' in f)
        calc_path    = next(os.path.join(data_dir, f) for f in files if 'Calcif' in f)

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
        self.adipose = adipose
        self.fibro   = fibro
        self.calc    = calc

        # Prepare per-pixel feature vectors
        # mu_low, mu_high from sinograms need reconstruction → skip for basic MLP
        # Here we demonstrate using mu arrays directly if pre-reconstructed.
        # For basic case, assume mu_low, mu_high already shape (N, H, W)
        if mu_low.ndim == 3:
            # flatten volumes
            vectors = np.stack([mu_low, mu_high], axis=-1)  # (N,H,W,2)
        else:
            raise ValueError('Expected mu_low to be 3D array')
        labels = np.zeros_like(adipose, dtype=np.int64)
        labels[adipose > 0.5] = 0
        labels[fibro   > 0.5] = 1
        labels[calc    > 0.5] = 2

        self.labels = labels
        self.vectors = vectors

        # Flatten to (num_pixels,)
        X = vectors.reshape(-1, 2)
        y = labels.reshape(-1)
        mask_valid = y >= 0
        X = X[mask_valid]
        y = y[mask_valid]

        N = X.shape[0]
        train_n = int(0.8 * N)
        mask = np.zeros(N, dtype=bool)
        mask[:train_n] = True
        np.random.seed(42)
        np.random.shuffle(mask)

        X_train      = X[mask]
        self.y_train = y[mask]
        X_test       = X[~mask]
        self.y_test  = y[~mask]

        # --- 4. Data scaling (standardization) ---
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

        '''
        print('============================')
        print('|          SHAPES          |')
        print('============================')
        print('X_train shape:    ', self.X_train.shape)
        print('y_train shape:    ', self.y_train.shape)
        print('Prototypes shape: ', np.array(prototypes).shape)
        print('----------------------------')
        '''

        self.train_ds = TensorDataset(
            torch.from_numpy(self.X_train),
            torch.from_numpy(self.y_train)
        )
        self.test_ds  = TensorDataset(
            torch.from_numpy(self.X_test),
            torch.from_numpy(self.y_test)
        )

    def subsample(self, subsample_size: float = 0.1):
        full_size_train = len(self.train_ds)
        subset_size = int(full_size_train * subsample_size)

        indices = torch.randperm(full_size_train)[:subset_size]
        train_subset = Subset(self.train_ds, indices)

        return DataLoader(train_subset, batch_size=65539, shuffle=True)


# Example usage:
# ds = MLPTestTrainDataset('./data')
# train_loader = DataLoader(ds.train_dataset, batch_size=4096, shuffle=True)
# test_loader  = DataLoader(ds.test_dataset,  batch_size=4096)
