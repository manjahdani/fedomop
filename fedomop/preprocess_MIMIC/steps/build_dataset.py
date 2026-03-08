import pandas as pd 
import numpy as np 
import pickle
import os 
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from joblib import Parallel, delayed

def build_dataset(use_ICU , label , disease_label , bucket , time  , oversampling , concat ):

    cohort_output = (
        "cohort_"
        + use_ICU.lower()
        + "_"
        + label.lower().replace(" ", "_")
        + "_"
        + str(time)
        + "_"
        + str(bucket)
        + "_"
        + disease_label
    )
    output_dir = './data/output/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + cohort_output
    os.makedirs(output_dir, exist_ok=True)

    hids = create_hids(oversampling)
    labels=pd.read_csv('./data/csv/labels.csv', header=0)
    concat_cols=[]
    if(concat):
        dyn=pd.read_csv('./data/csv/'+str(hids[0])+'/dynamic.csv',header=[0,1])
        dyn.columns=dyn.columns.droplevel(0)
        cols=dyn.columns
        time=dyn.shape[0]

        for t in range(time):
            cols_t = [x + "_"+str(t) for x in cols]

            concat_cols.extend(cols_t)
    X , Y = getXY(hids, labels , concat_cols , concat , use_ICU)
    print(X.shape)
    print("Saving dataset...")
    with open('./data/output/'+'X'+'.pkl', 'wb') as fp:
        pickle.dump(X, fp)
    with open('./data/output/'+'Y'+'.pkl', 'wb') as fp:
        pickle.dump(Y, fp)
    

def create_hids(oversampling):

    labels = pd.read_csv('./data/csv/labels.csv', header=0)

    hids = labels.iloc[:, 0]
    y = labels.iloc[:, 1]

    print("Total Samples", len(hids))
    print("Positive Samples", y.sum())

    if oversampling:
        print("=============OVERSAMPLING===============")
        oversample = RandomOverSampler(sampling_strategy='minority')
        hids = np.asarray(hids).reshape(-1, 1)
        hids, y = oversample.fit_resample(hids, y)
        hids = hids[:, 0]

        print("Total Samples", len(hids))
        print("Positive Samples", y.sum())
    
    return hids

def process_single_sample(sample, label_val, concat, concat_cols, data_icu, mean_keys, base_path):
    sample_dir = base_path / str(sample)
    
    # Read files
    dyn = pd.read_csv(sample_dir / 'dynamic.csv', header=[0, 1])
    stat = pd.read_csv(sample_dir / 'static.csv', header=[0, 1])['COND']
    demo = pd.read_csv(sample_dir / 'demo.csv')

    if concat:
        dyn_values = dyn.to_numpy().flatten().reshape(1, -1)
        dyn_df = pd.DataFrame(dyn_values, columns=concat_cols)
    else:
        agg_map = {col: ("mean" if col in mean_keys else "max") 
                   for col in dyn.columns.levels[0]}
        dyn_df = dyn.groupby(level=0, axis=1).agg(agg_map)
        dyn_df.columns = dyn_df.columns.droplevel(0)

    # Return the combined row and the label
    return pd.concat([dyn_df, stat, demo], axis=1), label_val

def getXY( ids, labels, concat_cols , concat , data_icu):

    X_rows = []

    y_rows = []


    if data_icu:

        label_map = labels.set_index('stay_id')['label']

    else:

        label_map = labels.set_index('hadm_id')['label']

    print(ids)
    for i, sample in enumerate(ids):

        if i % 100 == 0:
    
            print(i)


        y_rows.append(label_map.loc[sample])


        dyn = pd.read_csv(f'./data/csv/{sample}/dynamic.csv', header=[0,1])


        if concat:

            dyn.columns = dyn.columns.droplevel(0)

            dyn = dyn.to_numpy().reshape(1, -1)

            dyn_df = pd.DataFrame(dyn, columns=concat_cols)

        else:

            if data_icu:

                mean_keys = ["CHART", "MEDS"]

            else:

                mean_keys = ["LAB", "MEDS"]


            agg_map = {
                key: "mean" if key in mean_keys else "max"
                for key in dyn.columns.levels[0]

            }


            dyn_df = dyn.groupby(level=0, axis=1).agg(agg_map)

            dyn_df.columns = dyn_df.columns.droplevel(0)


        stat = pd.read_csv(f'./data/csv/{sample}/static.csv', header=[0,1])['COND']

        demo = pd.read_csv(f'./data/csv/{sample}/demo.csv')


        X_rows.append(pd.concat([dyn_df, stat, demo], axis=1))


    X_df = pd.concat(X_rows, axis=0, ignore_index=True)

    y_df = pd.Series(y_rows, name="label")


    return X_df, y_df 
