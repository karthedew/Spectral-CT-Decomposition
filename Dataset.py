# PyTorch Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AttnVectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return self.y.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TestTrainDataset:
    def __init__(self, mu_low_img, mu_high_img):

        # Stack attenuation vectors and flatten data
        vectors = np.stack([mu_low_img, mu_high_img], axis=-1)  # shape (N, H, W, 2)
        labels_arr = np.zeros_like(mu_low_img, dtype=np.int64)
        labels_arr[adipose > 0] = 0  # adipose class
        labels_arr[fibro > 0]   = 1  # fibroglandular class
        labels_arr[calc > 0]    = 2  # calcification class

        X = vectors.reshape(-1, 2)             # (N*H*W, 2)
        y = labels_arr.reshape(-1)             # (N*H*W,)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_ds = AttnVectorDataset(X_train, y_train)
        self.test_ds  = AttnVectorDataset(X_test,  y_test)
        self.train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=4096)
