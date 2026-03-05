import pickle
import numpy as np
import pandas as pd
 
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
#load X_train,Y_train,X_test,Y_test from Data/output/
import random
import os
 
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
 
seed_all(42)
 
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "/export/home/manjah/mimic-pfed/FedOMOP/preprocess_MIMIC/fold/"
print(f"Using {device}")
with open( path +'X_train_fold_0_small.pkl', 'rb') as fp:
    X_train = pickle.load(fp)
with open(path+'Y_train_fold_0_small.pkl', 'rb') as fp:
    Y_train = pickle.load(fp)
with open(path+'X_test_fold_0_small.pkl', 'rb') as fp:
    X_test = pickle.load(fp)
with open(path+'Y_test_fold_0_small.pkl', 'rb') as fp:
    Y_test = pickle.load(fp)
 
 
X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_array = np.concatenate([Y_train.to_numpy(), Y_test.to_numpy()]).astype(np.int64)
 
 
X_numeric = pd.get_dummies(X_all) # One-hot encode categorical columns
X_numeric = X_numeric.apply(pd.to_numeric, errors="coerce") # Force all columns to numeric
X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0) # Remove NaNs and infinities
 
X_array = X_numeric.to_numpy(dtype=np.float32)
 
print("X_shape", X_array.shape, "y_len", len(y_array))
 

fds = Dataset.from_dict({
    "features": X_array, # Keep as numpy
    "label": y_array,
})
# Set format to torch so the DataLoader returns Tensors, not lists
fds.set_format(type="torch", columns=["features", "label"])
 
fds.save_to_disk("fds_0")

from datasets import load_from_disk
fds = load_from_disk("/export/home/manjah/mimic-pfed/fds_0")
fds.set_format(type="torch", columns=["features", "label"])

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
 
    def forward(self, x):
        return x + self.block(x) # The Skip Connection
 
class FederatedResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.5):
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
 
 
def test(net, testloader, device):
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
 
 
def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_split_arg, seed : int):
    partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=dataset_split_arg,
                    min_partition_size=100,
                    self_balancing=True,
                    seed=seed)
    partitioner.dataset = fds
 
    client_dataset = partitioner.load_partition(partition_id)
 
    # Divide data on each node: 80% train, 20% validation
    partition_train_val = client_dataset.train_test_split(test_size=0.2, seed=seed)
    train_ds = partition_train_val["train"]
    val_ds  = partition_train_val["test"]
 
    trainloader = DataLoader(train_ds, 
                             batch_size=batch_size, 
                             shuffle=True)
    valloader  = DataLoader(val_ds,  
                            batch_size=batch_size, 
                            shuffle=False)
 
    return trainloader, valloader
 
 
trainl, vall = load_data(0, 1, 32, 1000, seed=42)
 
 
# # Extract all labels from the underlying dataset
# labels = torch.tensor(vall.dataset["label"]).detach().clone()
 
# num_ones = torch.sum(labels == 1).item()
# num_zeros = torch.sum(labels == 0).item()
# total = len(labels)
 
# print(f"Total Validation Samples: {total}")
# print(f"Readmissions (1): {num_ones} ({100 * num_ones/total:.2f}%)")
# print(f"No Readmission (0): {num_zeros} ({100 * num_zeros/total:.2f}%)")
 
 
# --- Train stats ---

# y_tr = trainl.dataset["label"]  # already a torch.Tensor
# n_tr = len(y_tr)


# n1_tr = (y_tr == 1).sum().item()
# n0_tr = (y_tr == 0).sum().item()
 
# y_va = vall.dataset["label"]  # already a torch.Tensor
# n_va = len(y_va)
# n1_va = (y_va == 1).sum().item()
# n0_va = (y_va == 0).sum().item()
# print("Samples:", f"Train: {n_tr}", f"Validation: {n_va}")
# print(f"Train Readmissions (1): {n1_tr} ({100*n1_tr/n_tr:.2f}%)", f"Validation Readmissions (1): {n1_va} ({100*n1_va/n_va:.2f}%)")
# print(f"Train No Readmission (0): {n0_tr} ({100*n0_tr/n_tr:.2f}%)", f"Validation No Readmission (0): {n0_va} ({100*n0_va/n_va:.2f}%)")
 
 
model = FederatedResNet(input_dim=X_array.shape[1])

print(X_array.shape[1])
epochs = 20
print(f"\n Training Resnet for {epochs} epochs")
train(model, trainl, epochs, 0.001, 0.9, 1e-4, device)
print(" Results loss, accuracy, auroc, auprc", test(model, vall, device))
 
 
print("\n Training Xgboost")
 
import xgboost as xgb
 
X_train_boost = np.concatenate([batch["features"].numpy() for batch in trainl]).astype(np.float32)
y_train_boost = np.concatenate([batch["label"].numpy() for batch in trainl])
 
X_test_boost = np.concatenate([batch["features"].numpy() for batch in vall]).astype(np.float32)
y_test_boost = np.concatenate([batch["label"].numpy() for batch in vall])
 
# 2. Train XGBoost (Standard clinical parameters)
# xgb_model = xgb.XGBClassifier(
#     n_estimators=1000,
#     max_depth=6,
#     random_state=42,
#     learning_rate=0.001,
#     tree_method="hist",  # Efficient for large MIMIC datasets
#     eval_metric='auc',
#     n_jobs=1,
#     early_stopping_rounds=50
# )
 
 
xgb_model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=1,
    random_state=42,
    early_stopping_rounds=100,
)
 
 
xgb_model.fit(
    X_train_boost, y_train_boost, 
    eval_set=[(X_test_boost, y_test_boost)], 
    verbose=False
)
 
y_probs_xgb = xgb_model.predict_proba(X_test_boost)[:, 1]
 
 
auc_xgb = roc_auc_score(y_test_boost, y_probs_xgb)
auprc_xgb = average_precision_score(y_test_boost, y_probs_xgb)
 
print("loss, accuracy, auroc, auprc", auc_xgb, auprc_xgb)