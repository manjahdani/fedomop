

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm.auto import tqdm

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x) # The Skip Connection

class FederatedResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.3):
        super().__init__()
        
        # --- SHARED BODY ---
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden_dim, dropout))
        self.body = nn.Sequential(*layers)
        
        # --- LOCAL HEAD ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Binary Classification
        )

    def forward(self, x):
        features = self.body(x)
        return self.head(features)



def train(net, trainloader, epochs, lr, momentum, weight_decay, device, reg_params=None, lamda=0.0):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    total_loss = 0.0

    for epoch in range(epochs):
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)  # already float [B,1] if you built it that way

            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)

            # FedProx proximal term: reg_params should be an iterable of tensors
            if reg_params is not None and lamda > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(net.parameters(), reg_params):
                    proximal_term += (local_p - global_p).pow(2).sum()
                loss = loss + (lamda / 2) * proximal_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

    
    return total_loss / (len(trainloader) * epochs)



def test(net, testloader, device):
    """Validate the model on the test set using clinical metrics."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    all_y_true = []
    all_y_probs = []

    net.eval()
    with torch.no_grad():
        for x, y in testloader:              # <-- tuple batch
            x = x.to(device)
            y = y.to(device)                 # already shape [B,1] float

            logits = net(x)
            total_loss += criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            all_y_true.append(y.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_probs = np.concatenate(all_y_probs, axis=0)

    loss = total_loss / len(testloader.dataset)
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)

    preds = (y_probs > 0.5).astype(np.int32)
    accuracy = (preds == y_true).mean()

    return loss, accuracy, auroc, auprc




def train_h(net, trainloader, epochs, lr, momentum, weight_decay, device, reg_params=None, lamda=0.0):
    net.to(device)
    # 1. BCE is standard for binary clinical tasks
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    net.train()
    total_loss = 0.0

    for epoch in range(epochs):
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1) # Reshape for BCE (N, 1)
            optimizer.zero_grad()
            
            # 3. Single forward pass
            logits = net(x)
            loss = criterion(logits, y)

            # 4. Proper FedProx implementation (Proximal term)
            if reg_params is not None and lamda > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(net.parameters(), reg_params):
                    proximal_term += (local_p - global_p).pow(2).sum()
                loss += (lamda / 2) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (len(trainloader) * epochs)



def test_h(net, testloader, device):
    """Validate the model on the test set using clinical metrics."""
    net.to(device)
    # Using 'sum' reduction for loss calculation across batches
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    
    total_loss = 0.0
    all_y_true = []
    all_y_probs = []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1)
            
            logits = net(x)
            total_loss += criterion(logits, y).item()
            
            # Convert logits to probabilities for AUROC/AUPRC
            probs = torch.sigmoid(logits)
            
            all_y_true.append(y.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_y_true)
    y_probs = np.concatenate(all_y_probs)

    # Calculate metrics
    loss = total_loss / len(testloader.dataset)
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    
    # Optional: Simple threshold at 0.5 for accuracy
    preds = (y_probs > 0.5).astype(int)
    accuracy = (preds == y_true).mean()

    return loss, accuracy, auroc, auprc


def _hospital_resnet(_f: int, _t: int):
    return FederatedResNet(input_dim=26, hidden_dim=128, n_blocks=3, dropout=0.5)

def _hospital_resnet_split(_f: int, _t: int):
    from maidam.ml.models.tabular_decomposable import ResnetSplit
    return ResnetSplit(FederatedResNet(input_dim=26, hidden_dim=128, n_blocks=3, dropout=0.5))

def _mimiciv_resnet(_f: int, _t: int):
    return FederatedResNet(input_dim=12818, hidden_dim=128, n_blocks=3, dropout=0.5)

def _mimiciv_resnet_split(_f: int, _t: int):
    from maidam.ml.models.tabular_decomposable import ResnetSplit
    return ResnetSplit(FederatedResNet(input_dim=12818, hidden_dim=128, n_blocks=3, dropout=0.5))