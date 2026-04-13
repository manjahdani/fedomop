
from collections import OrderedDict

from flwr.app import ArrayRecord, Context, Message, MetricRecord
import torch

from fedomop.helpers import save_layer_weights_to_state, load_layer_weights_from_state
from fedomop.task_utils import get_dataloaders
from fedomop.dataset import instantiate_ds_and_get_features
from fedomop.decomposable_model import ResnetManager
from torch.utils.data import DataLoader

def train_fedper(msg: Message, context: Context):
    
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    model_cls = context.run_config["model"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ---- build manager according to selected model ----
    
    trainloader, valloader = get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partitioner"], 
                                    context.run_config["dirichlet_alpha"])
    
    manager = _build_manager(
            model_cls,
            partition_id,
            dataset,
            trainloader,
            valloader,
            device= device,
        )
    

    # 0) restore client's private head (if previously saved)
    load_layer_weights_from_state(context.state, manager.model.head)

    # 1) sync body from server
    manager.model.set_body_from_ndarrays(msg.content["arrays"].to_torch_state_dict())
    

    # 2) local training (both body + head).
    train_metrics = manager.train(
            epochs=context.run_config["local-epochs"],
            lr=msg.content["config"]["lr"],
            weight_decay=context.run_config["weight_decay"],
            freeze_body=False,
            freeze_head=False,
        )

    # 3) save the private head for future rounds
    
    save_layer_weights_to_state(context.state, manager.model.head)
                    
    # 4) return only the BODY + training metrics
    
    full_state = OrderedDict()
    full_state.update(manager.model.body_state_dict())
    full_state.update(manager.model.head_state_dict())
    return full_state, train_metrics

    #return manager.model.get_body_parameters(), train_metrics

def eval_fedper(msg: Message, context: Context):
    """Evaluate the model on local data."""    
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    model_cls = context.run_config["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Load datasets
    trainloader, valloader = get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partitioner"], 
                                    context.run_config["dirichlet_alpha"])
    
    manager = _build_manager(
            model_cls,
            partition_id,
            dataset,
            trainloader,
            valloader,
            device= device,
        )

    load_layer_weights_from_state(context.state, manager.model.head)

    # 1) sync body from server
    manager.model.set_body_from_ndarrays(msg.content["arrays"].to_torch_state_dict())

    # Call the evaluation function
    
    return manager.test()

def _build_manager(model_name : str, 
                   client_id : int, 
                   dataset : str, 
                   trainloader: DataLoader, 
                   valloader: DataLoader, 
                   device: str):
    
    features = instantiate_ds_and_get_features(dataset=dataset)
    return ResnetManager(client_id=client_id, 
                         trainloader=trainloader, 
                         valloader=valloader, 
                         input_dim= len(features),
                         device=device)