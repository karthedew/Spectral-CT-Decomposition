# PyTorch Dataset and DataLoader
import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_npy_gz(path):
    '''
    Load a .npy.gz file and return the numpy array.
    '''
    with gzip.open(path, 'rb') as file:
        return np.load(file)


class AttnVectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return self.y.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        labels[adipose > 0] = 0
        labels[fibro   > 0] = 1
        labels[calc    > 0] = 2

        # Flatten to (num_pixels,)
        X = vectors.reshape(-1, 2)
        y = labels.reshape(-1)
        mask_valid = y >= 0
        X = X[mask_valid]
        y = y[mask_valid]

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Create PyTorch datasets
        self.train_dataset = AttnDataset(X_train, y_train)
        self.test_dataset  = AttnDataset(X_test,  y_test)


# Example usage:
# ds = MLPTestTrainDataset('./data')
# train_loader = DataLoader(ds.train_dataset, batch_size=4096, shuffle=True)
# test_loader  = DataLoader(ds.test_dataset,  batch_size=4096)
