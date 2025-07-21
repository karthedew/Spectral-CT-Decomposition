import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset
from multiprocessing import Pool, cpu_count

from src.fanbeam_fbp_sino import fbp  # your FBP function

def load_npy_gz(path: str) -> np.ndarray:
    """
    Load a gzipped .npy file and return its contents as a NumPy array.
    """
    with gzip.open(path, 'rb') as f:
        return np.load(f)


def _reconstruct_pair(mu_low_slice: np.ndarray, mu_high_slice: np.ndarray) -> np.ndarray:
    """
    Helper for parallel reconstruction: reconstructs low and high sinogram slices.
    Returns a (2, 512, 512) array.
    """
    low_img = fbp(mu_low_slice)
    high_img = fbp(mu_high_slice)
    return np.stack([low_img, high_img], axis=0)

class AttnDataset:
    """
    Loads or builds attenuation images for spectral CT:
      - Checks for a cached reconstruction of shape (N, 2, 512, 512).
      - If cache exists, loads it.
      - Otherwise, loads transmission, computes attenuation,
        reconstructs via FBP in parallel, and saves cache.

    Attributes:
        attn_images: np.ndarray of shape (N, 2, 512, 512)
        N: number of samples
    """
    def __init__(self, data_dir: str):
        # Paths
        cache_path = os.path.join(data_dir, 'attn_images.npy')

        # If cache exists, load and return
        if os.path.exists(cache_path):
            recon = np.load(cache_path)
        else:
            # Load raw transmission sinograms
            low_tx  = load_npy_gz('data/lowkVpTransmission.npy.gz')   # (N, 256, 1024)
            high_tx = load_npy_gz('data/highkVpTransmission.npy.gz')

            # Convert to attenuation (Beer-Lambert)
            eps = 1e-6
            mu_low  = -np.log(np.clip(low_tx,  eps, None))
            mu_high = -np.log(np.clip(high_tx, eps, None))

            # Parallel reconstruction via FBP
            N = mu_low.shape[0]
            with Pool(processes=cpu_count()-5) as pool:
                recon_list = pool.starmap(
                    _reconstruct_pair,
                    [(mu_low[i], mu_high[i]) for i in range(N)]
                )
            recon = np.stack(recon_list, axis=0).astype(np.float32)

            # Save for future use
            np.save(cache_path, recon)

        # Load ground-truth tissue maps
        self.images = recon
        self.labels = np.stack([
            load_npy_gz('data/Phantom_Adipose.npy.gz'),
            load_npy_gz('data/Phantom_Fibroglandular.npy.gz'),
            load_npy_gz('data/Phantom_Calcification.npy.gz')
        ], axis=1).astype(np.float32)  # (N, 3, 512, 512)

        self.dataset    = TissueSegmentationDataset(self.images, self.labels)
        self.train_size = int(0.8 * len(self.dataset))  # 800 samples
        self.test_size  = len(self.dataset) - self.train_size # 200 samples

        self.train_ds, self.test_ds = random_split(
            self.dataset,
            [self.train_size, self.test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def subsample(self, subsample_size: float = 0.01):
        full_size_train = len(self.train_ds)
        subset_size     = int(full_size_train * subsample_size)
        indices         = torch.randperm(full_size_train)[:subset_size]
        return Subset(self.train_ds, indices)


class TissueSegmentationDataset(Dataset):
    """
    PyTorch Dataset for spectral CT tissue segmentation.

    Wraps AttnDataset and pairs attenuation images with ground-truth masks.

    Usage:
        ds = TissueSegmentationDataset(data_dir, split='train')
        loader = DataLoader(ds, batch_size=8, shuffle=True)
    """
    def __init__(self, images, labels):
        self.x = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
