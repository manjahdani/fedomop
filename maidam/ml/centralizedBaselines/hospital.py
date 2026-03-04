import torch
import torch.nn as nn
import torch.nn.functional as F
from hopsital_dataset_utils import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
import numpy as np
import os
import joblib

SEED = 42
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
torch.use_deterministic_algorithms(True)  # raises if a non-deterministic op is used
torch.backends.cudnn.deterministic = True
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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



def main():
    EPOCHS = 1000
    LR = 1e-3
    WINDOW_DAYS = 30
    BASE_PATH = "../data/hospital/output_csv_cal/csv"  # Directory with your Synthea CSVs
    
    # -------------
    # MODELING
    # -------------
    out_prefix="any_readmit_30d"

    data = preprocessing (base=BASE_PATH, window_days=WINDOW_DAYS)
   

    y = data["readmit_flag"].astype(int)
    num_cols = ["age", "n_conditions_total", "prev_enc_180d"]
    cat_cols = [c for c in ["GENDER", "RACE", "ETHNICITY"] if c in data.columns]
    X = data[num_cols + cat_cols]
    print(X)
    print(num_cols + cat_cols)
    preprocess = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    log_reg = Pipeline([('prep', preprocess),
                        ('clf', LogisticRegression(max_iter=1000, n_jobs=1))])

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    log_reg.fit(X_train, y_train)
    
    results = pd.DataFrame([eval_model(log_reg, "LogisticRegression", X_test,y_test)])
    print(results)
    

    # -------------
    # SAVE OUTPUTS
    # -------------
    joblib.dump(log_reg, os.path.join(BASE_PATH, f"log_reg_{out_prefix}.pkl"))

    data[["Id", "PATIENT", "START", "STOP", "readmit_flag"]].to_csv(
        os.path.join(BASE_PATH, f"index_encounters_with_{out_prefix}_labels.csv"), index=False
    )
    results.to_csv(os.path.join(BASE_PATH, f"model_metrics_{out_prefix}.csv"), index=False)
    print("Done. Files saved in", BASE_PATH)
    
if __name__ == "__main__":
    main()