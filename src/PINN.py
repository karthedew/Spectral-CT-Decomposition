import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PinnWithPearsonMLP(nn.Module):
    """MLP that augments input with cosine similarities to class prototypes."""
    def __init__(self, prototypes: np.ndarray, hidden_dim: int = 64):
        super().__init__()
        self.prototypes = torch.tensor(prototypes)  # (C, 2)
        self.cosine = nn.CosineSimilarity(dim=1)
        input_dim = 2 + self.prototypes.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.prototypes.shape[0])
        )

    def forward(self, x):
        # x: (B,2)
        B = x.size(0)
        C = self.prototypes.size(0)
        p = self.prototypes.to(x.device)  # (C,2)
        x_exp = x.unsqueeze(1).expand(-1, C, -1).reshape(-1, 2)
        p_exp = p.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2)
        sims  = self.cosine(x_exp, p_exp).reshape(B, C)
        x_aug = torch.cat([x, sims], dim=1)
        return self.net(x_aug)


class CosMLPTrainer:
    """Handles training and evaluation of a model."""
    def __init__(self, train_dataset, test_dataset, prototypes,
                 batch_size: int = 4096, lr: float = 1e-3, device=None):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = PinnWithPearsonMLP(prototypes).to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.crit   = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for xb, yb, *_ in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss   = self.crit(logits, yb)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        return correct / total * 100

    def run(self, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            test_acc   = self.evaluate()
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Test Acc={test_acc:.2f}%")
