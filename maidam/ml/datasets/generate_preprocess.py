import json
import numpy as np
import pandas as pd

from hospital_dataset_utils import data_preparation

HOSPITALS_LARGE = ["parkcity","provo","sf","southbend","stlouis"]
HOSPITALS = ["arizona", "cal", "cct", "ohio", "mas", "oregon"]
WINDOW_DAYS = 30

def compute_schema(extra_features=("bmi","anxiety","medications","unit"), sample_per_hospital=None):
    cats_union = None
    num_cols_ref, cat_cols_ref = None, None

    for hid in range(len(HOSPITALS)):
        h_path = f"/export/home/manjah/data/hospital/output_csv_{HOSPITALS[hid]}/csv"
        _, X, y, _, num_cols, cat_cols = data_preparation(h_path, WINDOW_DAYS, list(extra_features))

        if sample_per_hospital is not None and len(X) > sample_per_hospital:
            X = X.sample(sample_per_hospital, random_state=42)

        if num_cols_ref is None:
            num_cols_ref, cat_cols_ref = num_cols, cat_cols
            cats_union = {c: set() for c in cat_cols_ref}

        for c in cat_cols_ref:
            if c in X.columns:
                cats_union[c].update(X[c].dropna().astype(str).unique().tolist())

    categories = {c: sorted(list(cats_union[c])) for c in cat_cols_ref}
    schema = {
        "num_cols": num_cols_ref,
        "cat_cols": cat_cols_ref,
        "categories": categories,
        "extra_features": list(extra_features),
    }
    return schema

schema = compute_schema(sample_per_hospital=5000)
with open("hospital_schema.json", "w") as f:
    json.dump(schema, f)