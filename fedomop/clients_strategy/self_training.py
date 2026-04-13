
from logging import INFO

from flwr.app import Context, Message
from flwr.common.logger import log
import torch

from fedomop.task_utils import create_instantiate_parameters, get_train_and_test_modules, get_dataloaders
from fedomop.helpers import save_model_from_to_state, load_model_from_state

personal_model_name = "personal_net"
def nn_model_training(net, dataset, model_cls, context: Context):

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    seed = context.run_config["seed"]
    batch_size = context.run_config["batch-size"]
    # Load Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_fn, _, _, _ = get_train_and_test_modules(dataset)

    trainloader, _ =  get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partitioner"], 
                                    context.run_config["dirichlet_alpha"])
    
    
    trainining_metrics = train_fn(
        net,
        trainloader,
        context.run_config["local-epochs"],
        context.run_config["lr"],
        context.run_config["weight_decay"],
        device)
    return trainining_metrics


def eval_self(msg: Message, context : Context):
    


    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]

    # Load the model and initialize it with the received weights
    

       # Load the data
    dataset = context.run_config["dataset"]

    # Load the model and initialize it with the received weights
    model_cls = context.run_config["model"]
    pmodel =  create_instantiate_parameters(dataset, model_cls)
    load_model_from_state(context.state, pmodel)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, valloader = get_dataloaders(dataset, 
                                    partition_id, 
                                    num_partitions, 
                                    batch_size, 
                                    seed, 
                                    context.run_config["partitioner"], 
                                    context.run_config["dirichlet_alpha"])

    _, test_fn, _, _ = get_train_and_test_modules(dataset)

    return test_fn(pmodel, valloader, device)

def train_self(msg: Message, context: Context):
    """Train the model on local data."""
    partition_id = context.node_config["partition-id"]
    dataset = context.run_config["dataset"]
    model_cls = context.run_config["model"]
    pmodel =  create_instantiate_parameters(dataset, model_cls)
    
    if personal_model_name not in context.state: 
        log(INFO, f"[Client {partition_id}] No Model found → training from scratch.")
        #Call the training function for personal 
        nn_model_training(pmodel, dataset, model_cls, context)
        save_model_from_to_state(context.state, pmodel)
    else:
        load_model_from_state(context.state, pmodel)
    

    #Call the training model
    training_metrics = nn_model_training(pmodel, dataset, model_cls, context)
        
    #Save Local in State Model
    save_model_from_to_state(context.state, pmodel)

    phantom_model = create_instantiate_parameters(dataset, model_cls) #@Fixme: this is a workaround to avoid sending
    return phantom_model.state_dict(), training_metrics