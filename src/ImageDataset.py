import numpy as np
import gzip
import torch
from torch.utils.data import Dataset, random_split, Subset


def load_npy_gz(path):
    '''
    Load a .npy.gz file and return the numpy array.
    '''
    with gzip.open(path, 'rb') as file:
        return np.load(file) #, mmap_mode='r')


class TissueSegmentationDataset(Dataset):

    def __init__(self, images, labels):
        self.x = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ImageDataset:

    def __init__(self):
        self.images = np.stack([
            load_npy_gz('data/highkVpImages.npy.gz'),
            load_npy_gz('data/lowkVpImages.npy.gz')
        ], axis=1)
        self.labels = np.stack([
            load_npy_gz('data/Phantom_Adipose.npy.gz'),
            load_npy_gz('data/Phantom_Fibroglandular.npy.gz'),
            load_npy_gz('data/Phantom_Calcification.npy.gz')
        ], axis=1)

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
