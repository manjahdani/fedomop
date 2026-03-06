from __future__ import annotations

from pathlib import Path
import os, random
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, cast
from collections import OrderedDict
from flwr.common.typing import UserConfig
import json
import datetime
from logging import INFO


import numpy as np

from flwr.common import (
    ConfigRecord,
    MetricRecord,
    RecordDict,
    log,
)

from fedomop.mimic_load import load_data_mimiiv, load_global_mimiiv
from fedomop.models.tabular import _mimiciv_resnet, _mimiciv_resnet_split
from fedomop.models.tabular_decomposable import ResnetManager


def seed_all(seed: int) -> None:
    # To ensure  

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic cuBLAS (PyTorch docs)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" for smaller
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Dataset registry ----------

@dataclass(frozen=True)
class DatasetSpec:
    features: Tuple[str, ...] | None
    targets: Tuple[str, ...] | None
    # factory takes (num_features, num_targets); it may ignore them
    models: Dict[str, Callable[[int, int], nn.Module]]
    criterion: str
    isErrorMetric: bool
    backend: str
    
    @property
    def num_features(self) -> int:
        return 0 if self.features is None else len(self.features)

    @property
    def num_targets(self) -> int:
        return 0 if self.targets is None else len(self.targets)

# ---------- manager factory ----------
def _build_manager(model_name : str, 
                   client_id : int, 
                   dataset : str, 
                   trainloader: DataLoader, 
                   valloader: DataLoader, 
                   device: str):
    


    
    
    return ResnetManager(client_id=client_id, 
                         trainloader=trainloader, 
                         valloader=valloader, 
                         input_dim= 12818,device=device)





def load_centralized_dataset(batch_size: int, seed: int):
    return load_global_mimiiv(batch_size, seed)


def _get_dataloaders(dataset: str, 
                     partition_id: int, 
                     num_partitions: int, 
                     batch_size: int, 
                     seed: int, 
                     partition_split: str, 
                     dataset_split_alpha: float):
    
    if dataset == "mimiciv":
        
        return load_data_mimiiv(partition_id, num_partitions, batch_size, dataset_split_alpha, seed)
    
    else:
        raise NotImplementedError(f"No method for {dataset}")



DATASETS: Dict[str, DatasetSpec] = {
    "mimiciv": DatasetSpec(
        features=None,
        targets= None,
        criterion="auroc",
        backend = "tabular",
        isErrorMetric = False,
        models={
            "ResnetMimic": _mimiciv_resnet,
            "ResnetSplitMimic": _mimiciv_resnet_split
        },
    ),}

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_train_and_test_modules(dataset: str):
    """Helper function to create a model."""

    spec = DATASETS[dataset]
    backend = getattr(spec, "backend")
    criterion = getattr(spec, "criterion")
    isErrorMetric = getattr(spec, "isErrorMetric")

    if backend == "tabular":
        from fedomop.models.tabular import train
        from fedomop.models.tabular import test

    else:
        raise NotImplementedError(f"No backend defined for dataset {dataset}")
    
    return train, test, isErrorMetric, criterion

def create_instantiate_parameters(dataset: str, model: str) -> nn.Module:
    """
    Create a model instance for the given dataset and model name.

    Raises:
        NotImplementedError: if dataset or model is unknown.
    """
    try:
        spec = DATASETS[dataset]
    except KeyError:
        available = ", ".join(sorted(DATASETS))
        raise NotImplementedError(f"Unknown dataset '{dataset}'. "
                                  f"Available: {available}") from None

    try:
        factory = spec.models[model]
    except KeyError:
        available = ", ".join(sorted(spec.models))
        raise NotImplementedError(f"Unfit model '{model}' for dataset '{dataset}'. "
                                  f"Available for '{dataset}': {available}") from None

    return factory(spec.num_features, spec.num_targets)


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def custom_aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors):
        for record_item in record.metric_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue
                if key not in aggregated_metrics:
                    if isinstance(value, list): #Treat the list case
                        aggregated_metrics[key] = [v * weight for v in value]
                    else:
                        aggregated_metrics[key] = value * weight #Treat the scalar case
                else:
                    if isinstance(value, list): #Treat the list case 
                        current_list = cast(list[float], aggregated_metrics[key])
                        aggregated_metrics[key] = [
                            curr + val * weight
                            for curr, val in zip(current_list, value)
                        ]
                    else:
                        current_value = cast(float, aggregated_metrics[key]) #Accumulate treat the scalar case
                        aggregated_metrics[key] = current_value + value * weight

    # Variance
    for record, weight in zip(records, weight_factors):
        for record_item in record.metric_records.values():
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    continue
                key_var_name = f'{key}_var'
                
                # 1. Calculate the squared difference (handles both scalar and list)
                diff_sq = weight * (np.array(value) - np.array(aggregated_metrics[key]))**2

                # 2. Update aggregated_metrics
                if key_var_name not in aggregated_metrics:
                    # Initialize: Remove the brackets [] to avoid the TypeError
                    aggregated_metrics[key_var_name] = diff_sq.tolist() if isinstance(value, list) else float(diff_sq)
                else:
                    if isinstance(value, list):
                        # Update list element-wise
                        current_var_list = cast(list[float], aggregated_metrics[key_var_name])
                        aggregated_metrics[key_var_name] = [curr + ds for curr, ds in zip(current_var_list, diff_sq)]
                    else:
                        # Update scalar
                        current_var_val = cast(float, aggregated_metrics[key_var_name])
                        aggregated_metrics[key_var_name] = current_var_val + float(diff_sq)

            # # --- MIN computation (unweighted across clients) ---
            # for key, value in record_item.items():
            #     if key == weighting_metric_name:
            #         continue

            #     key_min_name = f"{key}_min"

            #     if key_min_name not in aggregated_metrics:
            #         aggregated_metrics[key_min_name] = value
            #     else:
            #         if isinstance(value, list):
            #             current_min_list = cast(list[float], aggregated_metrics[key_min_name])
            #             aggregated_metrics[key_min_name] = [
            #                 min(curr, val) for curr, val in zip(current_min_list, value)
            #             ]
            #         else:
            #             current_min_val = cast(float, aggregated_metrics[key_min_name])
            #             aggregated_metrics[key_min_name] = min(current_min_val, value)


    return aggregated_metrics