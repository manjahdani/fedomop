from flwr.app import ArrayRecord, Context, Message, MetricRecord
import torch

from fedomop.task_utils import create_instantiate_parameters, get_train_and_test_modules, _get_dataloaders, get_weights
from fedomop.models.helpers import load_model_from_state, save_model_from_to_state

def train_ditto(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the data
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    model_cls = context.run_config["model"]
    seed = context.run_config["seed"]
    train_fn, _, _, _ = get_train_and_test_modules(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, _ = _get_dataloaders(dataset, 
                                      partition_id, 
                                      num_partitions, 
                                      batch_size, 
                                      seed, 
                                      context.run_config["partition_split"], 
                                      context.run_config["dataset_split_alpha"])

    # Load the model and initialize it with the received weights
    model = create_instantiate_parameters(dataset, model_cls)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    
    # Load personalized model

    pmodel = create_instantiate_parameters(dataset, model_cls)
    load_model_from_state(context.state, pmodel)
    
    
    reg_params = get_weights(model)
    train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            context.run_config["momentum"],
            context.run_config["weight_decay"],
            device,
        )

    ptrain_metrics = train_fn(
            pmodel,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            context.run_config["momentum"],
            context.run_config["weight_decay"],
            device,
            reg_params,
            context.run_config["ditto_lambda"]
        )

    save_model_from_to_state(context.state, pmodel)

    return model.state_dict(), ptrain_metrics

def eval_ditto(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]

    # Load the model and initialize it with the received weights
    model_cls = context.run_config["model"]
    
    pmodel = create_instantiate_parameters(dataset, model_cls) 
    load_model_from_state(context.state, pmodel)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   

    _, valloader = _get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partition_split"], 
                                    context.run_config["dataset_split_alpha"])

    _,test_fn,_, _ = get_train_and_test_modules(dataset)

    # Call the evaluation function
  
    return test_fn(pmodel, valloader,device)