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

fds = load_from_disk("/export/home/manjah/MIMIC-IV-Data-Pipeline/data/output/fds_fold0")
fds.set_format(type="torch", columns=["features", "label"])


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