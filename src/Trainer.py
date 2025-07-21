import torch
from torch import nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=8,
        lr=1e-3,
        loss_fn=None,
        optimizer=None,
        device=None,
        num_workers=4,
        verbose=False
    ):
        self.device       = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None
        self.crit         = loss_fn or nn.BCELoss()
        self.opt          = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)
        self.verbose      = verbose


    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self.model(xb)
            loss = self.crit(preds, yb)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss


    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        tissue_errors = torch.zeros(3, device=self.device)  # Adipose, Fibro, Calc
        count = 0

        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
        
                # Binary Cross-Entropy Loss (used for optimization)
                loss = self.crit(preds, yb)
                total_loss += loss.item()
        
                # Compute batch-level average tissue composition
                pred_pct_batch = preds.mean(dim=[2, 3])   # (B, 3)
                true_pct_batch = yb.mean(dim=[2, 3])      # (B, 3)
        
                # Accumulate absolute errors for each tissue
                tissue_errors += torch.sum(torch.abs(pred_pct_batch - true_pct_batch), dim=0)
                count += xb.size(0)
        
                # Visualization: pick first sample in batch
                pred_single = preds[0]   # (3, 512, 512)
                true_single = yb[0]      # (3, 512, 512)
        
                pred_pct = pred_single.mean(dim=(1, 2)).cpu().numpy() * 100
                true_pct = true_single.mean(dim=(1, 2)).cpu().numpy() * 100


        avg_loss = total_loss / len(self.val_loader)
        avg_error_pct = (tissue_errors / count).cpu().numpy()  # mean absolute error per tissue

        if self.verbose:
            print(f"           Val Loss:   {avg_loss:.4f}")
            print(f"Mean Absolute Error:  {avg_error_pct}")
            print("Tissue composition for a single image:")
            print(f"  Predicted:     Adipose: {pred_pct[0]:.2f}%, Fibro: {pred_pct[1]:.2f}%, Calc: {pred_pct[2]:.2f}%")
            print(f"  Ground Truth:  Adipose: {true_pct[0]:.2f}%, Fibro: {true_pct[1]:.2f}%, Calc: {true_pct[2]:.2f}%")
        return avg_loss


    def eval_percent(self):
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)

    def tune(self, hyperparameters: dict):
        filter_sizes = hyperparameters['filter_size']
        strides = hyperparameters['strides']

    def run(self, epochs=20):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            if self.verbose:
                print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f}")

            if self.val_loader:
                val_loss = self.evaluate()
                # print(f"           Val Loss:   {val_loss:.4f}")

            if self.verbose:
                print('---------------------')

