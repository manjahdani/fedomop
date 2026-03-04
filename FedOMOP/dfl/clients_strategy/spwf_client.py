from logging import INFO

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
import numpy as np
import torch

from maidam.ml.fl_common.task_utils import create_instantiate_parameters, get_train_and_test_modules, _get_dataloaders
from maidam.ml.models.helpers import save_model_from_to_state, load_model_from_state


def evaluate_peers(models_with_ids: list[tuple[int, ArrayRecord]], dataset, model_cls, context: Context):
    node_id = context.node_config["partition-id"]
    peer_models = {}
    peer_metrics = {}
    _, _, isErrorMetric, criterion = get_train_and_test_modules(dataset)

    for cid, arr_rec in models_with_ids:
        torch_state_dict = arr_rec.to_torch_state_dict()
        m = create_instantiate_parameters(dataset, model_cls)
        m.load_state_dict(torch_state_dict)  # now all values are torch.Tensor
        peer_models[cid]= m

        # PEER EVAL
        metrics = nn_eval(m, context)
        peer_metrics[cid]= metrics[f"eval_{criterion}"]
    
    adj_peer_metrics = {cid : x - peer_metrics[node_id] for cid, x in peer_metrics.items()}
    M = context.run_config["cooling_factor"]
    sign = -1.0 if isErrorMetric else 1.0
        
    z = {cid: np.exp(sign * M * v) for cid, v in adj_peer_metrics.items()}
    total = sum(z.values())
    weights = {cid : x/total for cid, x in z.items()}
    
    log(INFO, f"Client-id {node_id} - raw performance {peer_metrics}")
    log(INFO, f"Client-id {node_id} - adj performance {adj_peer_metrics}")
    log(INFO, f"Client-id {node_id} - weights {weights}")

    records : list[RecordDict] = []
    for cid, m in peer_models.items():
        records.append(RecordDict(
            {
                "arrays": ArrayRecord(m.state_dict()), 
                "metrics": MetricRecord({"softmax_w": weights[cid]})
                }))

    return aggregate_arrayrecords(records, "softmax_w")

def nn_eval(net, context : Context) -> ArrayRecord:
    
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]

    # Load the model and initialize it with the received weights
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, valloader = _get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partition_split"], 
                                    context.run_config["dataset_split_alpha"])

    _, test_fn, _, _ = get_train_and_test_modules(dataset)

    return test_fn(net, valloader, device)


def nn_model_training(net, dataset, model_cls, context: Context):

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    seed = context.run_config["seed"]
    batch_size = context.run_config["batch-size"]
    # Load Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_fn, _, _, _ = get_train_and_test_modules(dataset)

    trainloader, _ =  _get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partition_split"], 
                                    context.run_config["dataset_split_alpha"])

    training_metrics = train_fn(
        net,
        trainloader,
        context.run_config["local-epochs"],
        context.run_config["lr"],
        context.run_config["momentum"],
        context.
        run_config["weight_decay"],
        device)
    

    return training_metrics


def deserialize_weights(msg : Message)-> list[tuple[int, ArrayRecord]]:
    
    models_with_ids: list[tuple[int, ArrayRecord]] = []
    
    for key, rec in msg.content.items():
        if key.startswith("arrays_"):
            cid_str = key[len("arrays_"):]   # part after 'arrays_'
            cid = int(cid_str)               
            models_with_ids.append((cid, rec))

    return models_with_ids



def train_spwf(msg: Message, context: Context):
    """Train the model on local data."""

    dataset = context.run_config["dataset"]
    model_cls = context.run_config["model"]
    pmodel =  create_instantiate_parameters(dataset, model_cls)
    
    models_with_ids = deserialize_weights(msg)
    updated_model = evaluate_peers(models_with_ids, dataset, model_cls, context)
    pmodel.load_state_dict(updated_model.to_torch_state_dict())

    #Call the training model
    training_metrics = nn_model_training(pmodel, dataset, model_cls, context)
        
    #Save Local in State Model
    save_model_from_to_state(context.state, pmodel)

   

    return pmodel.state_dict(), training_metrics
   