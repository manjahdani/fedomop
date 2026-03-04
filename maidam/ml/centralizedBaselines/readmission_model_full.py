import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import train_test_split
import argparse

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from maidam.ml.models.tabular import FederatedResNet, train, test
from maidam.ml.datasets.hospital_dataset_utils import data_preparation, load_data, build_global_preprocessor
# ---------------------
# CONFIGURABLE SETTINGS
# ---------------------

HOSPITALS = ["arizona", "cal", "cct", "ohio", "mas", "oregon"]
HOSPITALS_LARGE = ["parkcity","provo","sf","southbend","stlouis"]
#HOSPITALS_LARGE = ["sf"]

WINDOW_DAYS = 30
BASE_PATH = "/export/home/manjah/Maidam/data/hospital/output_csv_mas/csv" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train_torch_model(model, X_tr, y_tr, X_te, y_te, lr=1e-3, epochs=20, batch_size=512, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32).view(-1, 1)

    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, drop_last=False)

    # pos_weight for imbalance: (neg/pos)
    pos = y_tr_t.sum().item()
    neg = y_tr_t.shape[0] - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    # predict proba on test
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device)).cpu().numpy().reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-logits))
    return model, prob





def eval_prob(prob, name, y_test):
    auc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    preds = (prob >= 0.5).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {"model": name, "AUROC": auc, "AUPRC": auprc,
            "precision": rep["1"]["precision"],
            "recall": rep["1"]["recall"],
            "f1": rep["1"]["f1-score"]}


def eval_model(model, name, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    preds = (prob >= 0.5).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {"model": name, "AUROC": auc, "AUPRC": auprc,
            "precision": rep["1"]["precision"],
            "recall": rep["1"]["recall"],
            "f1": rep["1"]["f1-score"]}

def main(base=BASE_PATH, extra_features = [], window_days=30, out_prefix="any_readmit_30d"):

    data, X, y, preprocessor, num_cols, cat_cols = data_preparation(base, WINDOW_DAYS, extra_features)

    log_reg = Pipeline([('prep', preprocessor),
                        ('clf', LogisticRegression(max_iter=3000, n_jobs=1))])
    
    rf = Pipeline([('prep', preprocessor),
                   ('clf', RandomForestClassifier(n_estimators=300, random_state=42))])

    gdb = Pipeline([('prep', preprocessor),
                   ('clf', GradientBoostingClassifier(n_estimators=300, random_state=42))])
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    X_transformed = preprocessor.fit_transform(X_train)
        # Access the OneHotEncoder inside
    
    ohe = preprocessor.named_transformers_['cat']
    # Get feature names for categorical columns
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    all_features_names = num_cols + list(cat_feature_names)
    
    MI = mutual_info_classif(X_transformed, y_train.to_numpy(), n_neighbors=3, random_state=42)
    mi_series = pd.Series(MI, index=all_features_names)
    # Sort descending
    mi_sorted = mi_series.sort_values(ascending=False)
    print(mi_sorted)

        # Fit preprocessor on train, transform train/test
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp  = preprocessor.transform(X_test)

    # Densify only the transformed matrices (often sparse because of OneHotEncoder)
    if hasattr(X_train_pp, "toarray"):
        X_train_dense = X_train_pp.toarray().astype(np.float32)
        X_test_dense  = X_test_pp.toarray().astype(np.float32)
    else:
        X_train_dense = np.asarray(X_train_pp, dtype=np.float32)
        X_test_dense  = np.asarray(X_test_pp, dtype=np.float32)

    # train torch model
    input_dim = X_train_dense.shape[1]
    torch_model = FederatedResNet(input_dim=input_dim, hidden_dim=128, n_blocks=3, dropout=0.5)

    torch_model, prob_nn = train_torch_model(
        torch_model,
        X_train_dense, y_train.to_numpy(),
        X_test_dense,  y_test.to_numpy(),
        lr=1e-3, epochs=50, batch_size=512
    )


    

    print(eval_prob(prob_nn, "FederatedResNet", y_test.to_numpy()))

    #MODEL FITTING
    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gdb.fit(X_train, y_train)

    results = pd.DataFrame([eval_model(log_reg, "LogisticRegression", X_test, y_test),
                            eval_model(rf, "RandomForest", X_test, y_test),
                            eval_model(gdb, "XgBoost", X_test, y_test)])
    print(results)

    # -------------
    # SAVE OUTPUTS
    # -------------
    joblib.dump(log_reg, os.path.join(base, f"log_reg_{out_prefix}.pkl"))
    joblib.dump(rf, os.path.join(base, f"rf_{out_prefix}.pkl"))
    data[["Id", "PATIENT", "START", "STOP", "readmit_flag"]].to_csv(
        os.path.join(base, f"index_encounters_with_{out_prefix}_labels.csv"), index=False
    )
    results.to_csv(os.path.join(base, f"model_metrics_{out_prefix}.csv"), index=False)
    print("Done. Files saved in", base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes a list of items")
    parser.add_argument("items", nargs="+",  help="List of items",  default = [])
    args = parser.parse_args()
    #for h in HOSPITALS[0]:
     #   print(f"Analyzing hospital {h}")
        #h_path = f"/export/home/manjah/data/hospital/large_ds/output_{h}_large/csv"
    #h_path = f"/export/home/manjah/data/hospital/large_ds/output_parkcity_large/csv"
    #main(base=h_path, extra_features = args.items)

        # train torch model
    
    # preprocessor, num_cols, cat_cols = build_global_preprocessor()
    # for i in range(5):
    #     train_dl, val_dl = load_data(i, preprocessor, num_cols, cat_cols, batch_size=128)
    #     x0, y0 = next(iter(train_dl))
    #     input_dim = x0.shape[1]
    #     print("Input_dim", input_dim)
    #     torch_model = FederatedResNet(input_dim=input_dim, hidden_dim=128, n_blocks=3, dropout=0.5)
    #     print("Training")
    #     train(torch_model, train_dl, 50, 0.001, 0.9, 1e-4, device)
    #     print("Evaluating")
    #     print(test(torch_model, val_dl, device))

    from torch.utils.data import ConcatDataset, DataLoader

    preprocessor, num_cols, cat_cols = build_global_preprocessor()

    train_sets = []
    val_sets = []

    for i in range(6):
        train_dl, val_dl = load_data(i, preprocessor, num_cols, cat_cols, batch_size=128)
        train_sets.append(train_dl.dataset)
        val_sets.append(val_dl.dataset)

    train_ds_all = ConcatDataset(train_sets)
    val_ds_all   = ConcatDataset(val_sets)

    train_dl_all = DataLoader(train_ds_all, batch_size=128, shuffle=True, num_workers=0)
    val_dl_all   = DataLoader(val_ds_all,   batch_size=128, shuffle=False, num_workers=0)

    # sanity check
    x0, y0 = next(iter(train_dl_all))
    input_dim = x0.shape[1]

    torch_model = FederatedResNet(input_dim=input_dim, hidden_dim=128, n_blocks=3, dropout=0.5)
    train(torch_model, train_dl_all, 100, 0.001, 0.9, 1e-4, device)
    print(test(torch_model, val_dl_all, device))

   
