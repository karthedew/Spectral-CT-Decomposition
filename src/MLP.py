import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BasicMLP(nn.Module):
    """MLP that augments input with cosine similarities to class prototypes."""
    def __init__(self):
        super().__init__()
        input_dim = 2 * 256 * 1024  # Flattened input

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def pearson_corr(self, x, y):
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        cov = (x_centered * y_centered).mean(dim=1)
        std_x = x.std(dim=1)
        std_y = y.std(dim=1)
        return cov / (std_x * std_y + 1e-8)


class MLPTrainer:
    """Handles training and evaluation of a model."""
    def __init__(
        self,
        train_dataset,
        test_dataset,
        batch_size: int = 65539,
        alpha: float = 1e-2,
        device=None
    ):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = BasicMLP().to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.alpha  = alpha
        self.crit   = nn.CrossEntropyLoss()

    def calculate_loss(self, logits, x_batch, prototypes):
        probs = F.softmax(logits, dim=1)
        mu_pred = probs @ prototypes
        return F.mse_loss(mu_pred, x_batch)


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
        all_predictions = []
        all_labels      = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                print('Test xb shape: ', xb.shape)
                print('Test yb shape: ', yb.shape)
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb) # .argmax(dim=1)
                # Convert the predictions to 0 or 1
                preds = (logits >= 0.5).float()
                #print('Predictions shape: ', preds.shape)
                #print('Prediction 0: ', preds[0])
                #print('Preds: ', preds)
                #print('Actual: ', yb)
                all_predictions.append(preds.cpu())
                all_labels.append(yb.cpu().int())

        preds = torch.cat(all_predictions)
        labels = torch.cat(all_labels)

        per_class_acc = self._evaluate_per_class_accuracy(labels, preds)
        exact_match_acc = self._evaluate_exact_match(labels, preds)
        # accuracy_per_class = correct_per_class.float() / total_per_class

        return per_class_acc, exact_match_acc


    def _evaluate_per_class_accuracy(self, yb, preds):
        # Both yb and preds should be torch tensors of shape (N, 3)
        correct_per_class = (preds == yb).sum(dim=0)         # shape: (3,)
        print('Correct per class: ', correct_per_class)
        total_per_class   = torch.ones_like(yb).sum(dim=0)   # shape: (3,)
        print('Total Per class: ', total_per_class)
        adipose_acc = correct_per_class[0] / total_per_class[0] * 100
        fibro_acc   = correct_per_class[1] / total_per_class[1] * 100
        calc_acc    = correct_per_class[2] / total_per_class[2] * 100
        return (adipose_acc, fibro_acc, calc_acc)


    def _evaluate_exact_match(self, yb, preds):
        correct_samples = (preds == yb).all(dim=1).sum()
        accuracy = correct_samples.float() / yb.shape[0]
        return accuracy

    def run(self, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            per_class_acc, exact_match_acc   = self.evaluate()
            #print('Test Accuracy: \t', test_acc)
            print('++++++++++++++++++++++++++++')
            print(per_class_acc)
            print(exact_match_acc)
            print('++++++++++++++++++++++++++++')
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Class Acc={per_class_acc:.2f}%, Exact Match Acc={exact_match_acc:.2f}%")
