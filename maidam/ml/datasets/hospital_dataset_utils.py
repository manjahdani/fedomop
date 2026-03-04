import pandas as pd 
import os 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

from flwr_datasets.partitioner import DirichletPartitioner
from datasets import load_from_disk
HOSPITALS_LARGE = ["parkcity","provo","sf","southbend","stlouis"]
HOSPITALS = ["arizona", "cal", "cct", "ohio", "mas", "oregon"]
WINDOW_DAYS = 30


fds = load_from_disk("/export/home/manjah/MIMIC-IV-Data-Pipeline/data/output/fds_fold0")
fds.set_format(type="torch", columns=["features", "label"])

def to_dt(series):
    """Parse datetimes, remove timezone if present."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s
    # Previous encounters in past 180 days

def build_any_encounter_readmit(enc, window_days=30):
    """For each encounter discharge, flag if next encounter (any type) within window_days."""
    enc = enc.copy()
    enc["START"] = to_dt(enc["START"])
    enc["STOP"] = to_dt(enc["STOP"])
    enc = enc.sort_values(["PATIENT", "START"]).reset_index(drop=True)
    enc["readmit_flag"] = 0
    for pid, grp in enc.groupby("PATIENT"):
        grp = grp.sort_values("START")
        starts = grp["START"].values
        stops = grp["STOP"].values
        idxs = grp.index.values
        for i in range(len(grp)):
            discharge = stops[i]
            mask = (starts > discharge) & (starts <= discharge + np.timedelta64(window_days, 'D'))
            if mask.any():
                enc.loc[idxs[i], "readmit_flag"] = 1
    return enc

def patient_demographics(base, data):
    pat = pd.read_csv(os.path.join(base, "patients.csv"))
    
    pat["BIRTHDATE"] = to_dt(pat["BIRTHDATE"])
    data = data.merge(
        pat.rename(columns={"Id": "PATIENT"})[["PATIENT", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]],
        on="PATIENT", how="left"
    )
    data["age"] = (to_dt(data["STOP"]) - data["BIRTHDATE"]).dt.days / 365.25
    return data

def comorbidity(base, data):
    cond = pd.read_csv(os.path.join(base, "conditions.csv"))
        
    cond["START"] = to_dt(cond["START"])
    cond_grp = cond.groupby("PATIENT")["CODE"].nunique().rename("n_conditions_total")
    data = data.merge(cond_grp, on="PATIENT", how="left")
    return data


def vital_signs_bmi(base, data):
    obs = pd.read_csv(os.path.join(base,"observations.csv"))
    obs["DATE"] = to_dt(obs["DATE"])
    bmi = obs[
        obs["DESCRIPTION"].astype(str).str.contains("body mass index", case=False, na=False)
        | (obs.get("CODE", pd.Series("", index=obs.index)).astype(str) == "39156-5")
    ][["PATIENT", "DATE", "VALUE"]].copy()

    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi = bmi.dropna(subset=["VALUE"]).rename(columns={"VALUE": "vs_bmi"})

    # --- manual asof: for each (PATIENT, START), take the last BMI with DATE <= START ---
    pair = (
        data[["PATIENT", "START"]]
        .merge(bmi, on="PATIENT", how="left")
    )
    pair = pair[pair["DATE"] <= pair["START"]]
    pair = pair.sort_values(["PATIENT", "START", "DATE"])
    pair_last = pair.drop_duplicates(subset=["PATIENT", "START"], keep="last")[["PATIENT", "START", "vs_bmi"]]

    # attach vs_bmi back
    data = data.merge(pair_last, on=["PATIENT", "START"], how="left")
    return data, ["vs_bmi"]

def vital_signs_anxiety(base, data):
    obs = pd.read_csv(os.path.join(base,"observations.csv"))
    obs["DATE"] = to_dt(obs["DATE"])
    
    #--- GAD-7 total score → anxiety_selfreported (0–21) ---
    gad = obs[
        obs["DESCRIPTION"].astype(str).str.contains("gad-7|generalized anxiety disorder 7", case=False, na=False)
    ][["PATIENT", "DATE", "VALUE"]].copy()

    gad["VALUE"] = pd.to_numeric(gad["VALUE"], errors="coerce")
    gad = gad.dropna(subset=["VALUE"]).rename(columns={"VALUE": "anxiety_selfreported"})

    # manual asof for GAD-7: last score with DATE <= START
    pair_g = data[["PATIENT", "START"]].merge(gad, on="PATIENT", how="left")
    pair_g = pair_g[pair_g["DATE"] <= pair_g["START"]].sort_values(["PATIENT","START","DATE"])
    pair_g_last = pair_g.drop_duplicates(subset=["PATIENT","START"], keep="last")[["PATIENT","START","anxiety_selfreported"]]

    data = data.merge(pair_g_last, on=["PATIENT","START"], how="left")

    # clip to valid GAD-7 range
    data["anxiety_selfreported"] = pd.to_numeric(data["anxiety_selfreported"], errors="coerce").clip(0, 21)
    return data, ["anxiety_selfreported"]


def medications_features(base, data, lookback_days=180):
    meds = pd.read_csv(os.path.join(base, "medications.csv"))
    meds["START"] = to_dt(meds["START"])
    meds["STOP"] = to_dt(meds["STOP"])
    
    # Keep only useful columns
    meds = meds[["PATIENT","START","STOP","CODE","TOTALCOST"]].copy()
    meds["TOTALCOST"] = pd.to_numeric(meds["TOTALCOST"], errors="coerce").fillna(0)

    # Pre-index container
    data["meds_count_180d"] = 0
    data["meds_totalcost_180d"] = 0.0
    
    # Group meds by patient for speed
    meds_by_pid = {pid:g for pid,g in meds.groupby("PATIENT", sort=False)}

    def _fill_meds(row):
        pid, st = row["PATIENT"], row["START"]
        g = meds_by_pid.get(pid)
        if g is None or g.empty:
            return row
        win = g[(g["STOP"] < st) & (g["STOP"] >= st - pd.Timedelta(days=lookback_days))]
        if not win.empty:
            row["meds_count_180d"] = win["CODE"].nunique()
            row["meds_totalcost_180d"] = win["TOTALCOST"].sum()
        return row
    
    data = data.apply(_fill_meds, axis=1)
    return data, ["meds_count_180d", "meds_totalcost_180d"]


def treatment_unit(base, data):
    prov = pd.read_csv(os.path.join(base, "providers.csv"))
    org = pd.read_csv(os.path.join(base, "organizations.csv"))

    prov = prov.rename(columns={"Id": "PROVIDER_ID"})
    org = org.rename(columns={"Id": "ORG_ID"})

    # Try to find an org type column
    possible_type_cols = [c for c in org.columns if c.lower() in ["type", "organization_type", "resource_type"]]
    if possible_type_cols:
        type_col = possible_type_cols[0]
        org = org.rename(columns={type_col: "ORG_TYPE"})
    else:
        #print("WARNING: No TYPE column found in organizations.csv, skipping ORG_TYPE")
        org["ORG_TYPE"] = np.nan

    # Merge encounters → providers
    data = data.merge(
        prov[["PROVIDER_ID", "SPECIALITY", "ORGANIZATION"]].rename(columns={"ORGANIZATION": "ORG_ID"}),
        left_on="PROVIDER",
        right_on="PROVIDER_ID",
        how="left"
    )

    # Merge providers → organizations
    data = data.merge(org[["ORG_ID", "ORG_TYPE"]], on="ORG_ID", how="left")

    data = data.drop(columns=["PROVIDER_ID", "ORG_ID"], errors="ignore")

    for c in ["SPECIALITY", "ORG_TYPE", "ENCOUNTERCLASS"]:
        if c in data.columns:
            data[c] = data[c].astype("object")

    #print("SPECIALITY:", data["SPECIALITY"].dropna().unique()[:20])
    #print("ORG_TYPE:", data["ORG_TYPE"].dropna().unique())
    #print("ENCOUNTERCLASS:", data["ENCOUNTERCLASS"].dropna().unique())
    return data, ["SPECIALITY", "ORG_TYPE", "ENCOUNTERCLASS"]



def data_preparation(base, window_days, extra_features=[]):
    # -------------
    # LOAD DATA
    # -------------
    enc = pd.read_csv(os.path.join(base, "encounters.csv"))
    
    # -------------
    # LABEL LOGIC
    # -------------
    data = build_any_encounter_readmit(enc, window_days)
    print("Label counts (readmit_flag):", np.bincount(data["readmit_flag"]))
    
    # -------------
    # FEATURE ENGINEERING
    # -------------
    
    # Patient demographics
    data = patient_demographics(base, data)

    # Comorbidity count (distinct conditions ever for this patient)
    data = comorbidity(base, data)
    
    
    def prev_count(row):
        pid = row["PATIENT"]
        st = row["START"]
        return ((data["PATIENT"] == pid) &
                (data["STOP"] < st) &
                (data["STOP"] >= st - pd.Timedelta(days=180))).sum()

    data["prev_enc_180d"] = data.apply(prev_count, axis=1)

    num_cols = ["age", "n_conditions_total", "prev_enc_180d"]
    meta_cat_cols = ["GENDER", "RACE", "ETHNICITY"]
    # Vital Signs
    
    for ef in extra_features:
        if ef == "bmi":
            data, num_col_vs = vital_signs_bmi(base, data)
            num_cols.extend(num_col_vs)
        elif ef == "anxiety":
            data, num_col_vs = vital_signs_anxiety(base, data)
            num_cols.extend(num_col_vs)
        elif ef == "medications":
            data, num_col_vs = medications_features(base, data)
            num_cols.extend(num_col_vs)
        elif ef == "unit":
            data, cat_cols_vs = treatment_unit(base, data)
            # extend categorical columns
            meta_cat_cols.extend(cat_cols_vs)
        else:
            raise NotImplementedError("Unknown extra featues")
    
    # -------------
    # DATA PREPROCESSING
    # -------------

    #Convert Numeric and Fill NA 
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].fillna(data[c].median())
    
    # -------------
    # MODELING
    # -------------
    y = data["readmit_flag"].astype(int)
    #num_cols = ["age", "n_conditions_total", "prev_enc_180d"]
    cat_cols = [c for c in meta_cat_cols if c in data.columns]

    
    X = data[num_cols + cat_cols]

    preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scale', StandardScaler())   
    ]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

    return data, X, y, preprocessor, num_cols, cat_cols


def load_schema(path="hospital_schema.json"):
    import json
    with open(path, "r") as f:
        return json.load(f)
    
def build_global_preprocessor():
    schema = load_schema("/export/home/manjah/Maidam/maidam/ml/datasets/hospital_schema.json")

    num_cols = schema["num_cols"]

    cat_cols_filtered = []
    categories_filtered = []

    for c in schema["cat_cols"]:
        cats = schema["categories"].get(c, [])
        if cats is not None and len(cats) > 0:
            cat_cols_filtered.append(c)
            categories_filtered.append(cats)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", categories=categories_filtered), cat_cols_filtered),
    ])

    return preprocessor, num_cols, cat_cols_filtered



def load_data(partition_id: int, preprocessor, num_cols, cat_cols, batch_size: int = 128,
              extra_features=("bmi","anxiety","medications","unit")) -> DataLoader:

    h_path = f"/export/home/manjah/data/hospital/output_csv_{HOSPITALS[partition_id]}/csv"
    _, X, y, _, _, _ = data_preparation(h_path, WINDOW_DAYS, list(extra_features))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # IMPORTANT: do NOT fit here
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp  = preprocessor.transform(X_test)

    # (optional) feature names (available because global preprocessor was fit)
    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
    else:
        cat_feature_names = []

    # Densify
    if hasattr(X_train_pp, "toarray"):
        X_train_dense = X_train_pp.toarray().astype(np.float32)
        X_test_dense  = X_test_pp.toarray().astype(np.float32)
    else:
        X_train_dense = np.asarray(X_train_pp, dtype=np.float32)
        X_test_dense  = np.asarray(X_test_pp, dtype=np.float32)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np  = y_test.to_numpy(dtype=np.float32)

    X_tr_t = torch.from_numpy(X_train_dense).float()
    y_tr_t = torch.from_numpy(y_train_np).view(-1, 1)

    X_te_t = torch.from_numpy(X_test_dense).float()
    y_te_t = torch.from_numpy(y_test_np).view(-1, 1)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds  = TensorDataset(X_te_t, y_te_t)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valloader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return trainloader, valloader




def load_data_mimiiv(partition_id: int, num_partitions: int, batch_size: int, dataset_split_arg, seed : int):
    
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








# def load_data(partition_id: int, batch_size: int = 128):
    
#     #h_path = f"/export/home/manjah/data/hospital/large_ds/output_{HOSPITALS_LARGE[partition_id]}_large/csv"

#     h_path = f"/export/home/manjah/data/hospital/output_csv_{HOSPITALS[partition_id]}/csv"
#     _, X, y, preprocessor, num_cols, cat_cols = data_preparation(
#         h_path, WINDOW_DAYS, ["bmi","anxiety","medications","unit"]
#     )

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, stratify=y, random_state=42
#     )

#     # Fit on train once
#     X_train_pp = preprocessor.fit_transform(X_train)
#     X_test_pp  = preprocessor.transform(X_test)

#     # Optional: feature names (only after fit)
#     if "cat" in preprocessor.named_transformers_ and len(cat_cols) > 0:
#         ohe = preprocessor.named_transformers_["cat"]
#         cat_feature_names = ohe.get_feature_names_out(cat_cols)
#     else:
#         cat_feature_names = []

#     # Densify
#     if hasattr(X_train_pp, "toarray"):
#         X_train_dense = X_train_pp.toarray().astype(np.float32)
#         X_test_dense  = X_test_pp.toarray().astype(np.float32)
#     else:
#         X_train_dense = np.asarray(X_train_pp, dtype=np.float32)
#         X_test_dense  = np.asarray(X_test_pp, dtype=np.float32)

#     # y: make it numpy first (fixes your error)
#     y_train_np = y_train.to_numpy(dtype=np.float32)
#     y_test_np  = y_test.to_numpy(dtype=np.float32)

#     X_tr_t = torch.from_numpy(X_train_dense).float()
#     y_tr_t = torch.from_numpy(y_train_np).view(-1, 1)

#     X_te_t = torch.from_numpy(X_test_dense).float()
#     y_te_t = torch.from_numpy(y_test_np).view(-1, 1)

#     train_ds = TensorDataset(X_tr_t, y_tr_t)
#     test_ds  = TensorDataset(X_te_t, y_te_t)

#     trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
#     valloader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

#     return trainloader, valloader





# def build_global_preprocessor(extra_features=("bmi","anxiety","medications","unit")):
#     # Build X for all hospitals and pool them (can sample if too big)
#     X_list = []
#     num_cols_ref, cat_cols_ref = None, None

#     for hid in range(len(HOSPITALS)):
#         h_path = f"/export/home/manjah/data/hospital/output_csv_{HOSPITALS[hid]}/csv"
#         _, X, y, _, num_cols, cat_cols = data_preparation(h_path, WINDOW_DAYS, list(extra_features))
#         X_list.append(X)

#         if num_cols_ref is None:
#             num_cols_ref, cat_cols_ref = num_cols, cat_cols

#     X_all = pd.concat(X_list, axis=0, ignore_index=True)

#     preprocessor = ColumnTransformer([
#         ("num", Pipeline([("scale", StandardScaler())]), num_cols_ref),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_ref),
#     ])

#     preprocessor.fit(X_all)
#     return preprocessor, num_cols_ref, cat_cols_ref