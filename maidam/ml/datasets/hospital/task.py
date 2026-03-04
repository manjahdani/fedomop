import numpy as np
from flwr.common import NDArrays
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os 
import pandas as pd
BASE_PATH = "../data/hospital"
HOSPITALS = ["arizona", "cal", "cct", "ohio", "mas", "oregon"]
# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS = [0, 1]
N_FEATURES = None
SHARED_CATEGORIES = None
CAT_COLS = None

def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def create_log_reg_and_instantiate_parameters(penalty):
    global SHARED_CATEGORIES, CAT_COLS, N_FEATURES

    if N_FEATURES is None:
        a,b,c,ds = load_data(0, len(HOSPITALS))
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1000,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
        solver="saga",
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=N_FEATURES, n_classes=len(UNIQUE_LABELS))
    return model

# Ajouter les départements, et spécialités du médecin comme feature. 

def gather_shared_categories():
    all_cats = defaultdict(set)
    for hosp in HOSPITALS:
        hosp_path = os.path.join(BASE_PATH, f"output_csv_{hosp}", "csv")
        pat = pd.read_csv(os.path.join(hosp_path, "patients.csv"))
        for col in ["GENDER", "RACE", "ETHNICITY"]:
            if col in pat.columns:
                all_cats[col].update(pat[col].dropna().unique())
    cat_cols = [c for c in ["GENDER", "RACE", "ETHNICITY"] if all_cats[c]]
    shared_categories = [sorted(all_cats[c]) for c in cat_cols]
    return shared_categories, cat_cols


def load_data(partition_id: int, num_partitions: int):
    global SHARED_CATEGORIES, CAT_COLS, N_FEATURES

    if SHARED_CATEGORIES is None or CAT_COLS is None:
        SHARED_CATEGORIES, CAT_COLS = gather_shared_categories()
    EPOCHS = 1000
    LR = 1e-3
    WINDOW_DAYS = 30
    BASE_PATH = f"../data/hospital/output_csv_{HOSPITALS[partition_id]}/csv"  # Directory with your Synthea CSVs
    
    # -------------
    # MODELING
    # -------------
    out_prefix="any_readmit_30d"

    data = preprocessing(base=BASE_PATH, window_days=WINDOW_DAYS)
   

    y = data["readmit_flag"].astype(int)
    num_cols = ["age", "n_conditions_total", "prev_enc_180d"]
    
    X = data[num_cols + CAT_COLS]
    
    preprocess = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(categories=SHARED_CATEGORIES, handle_unknown='ignore'), CAT_COLS)
    ])
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    X_train_enc = preprocess.fit_transform(X_train)
    X_test_enc = preprocess.transform(X_test)

    if N_FEATURES is None:
        N_FEATURES = X_train_enc.shape[1]
    return X_train_enc, y_train, X_test_enc, y_test