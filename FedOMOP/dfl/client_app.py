from logging import INFO
from typing import Callable, Tuple

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from FedOMOP.dfl.clients_strategy.self_training_client import train_self
from FedOMOP.dfl.clients_strategy.spwf_client import nn_model_training, train_spwf, nn_eval
from FedOMOP.ml.fl_common.task_utils import create_instantiate_parameters
from FedOMOP.ml.models.helpers import load_model_from_state, save_model_from_to_state
from FedOMOP.ml.fl_common.task_utils import seed_all

# Flower ClientApp
app = ClientApp()


TrainHandler = Callable[[Message, Context], Tuple[ArrayRecord, MetricRecord]]
EvaluateHandler =  Callable[[Message, Context], Tuple[ArrayRecord, MetricRecord]]

TRAIN_HANDLERS: dict[str, TrainHandler] = {
    "SPWF": train_spwf,
    "SOLO": train_self
}

personal_model_name = "personal_net"

@app.query()
def send_weight(msg: Message, context: Context):
    """Train the model on local data."""
    seed_all(context.run_config["seed"])
    partition_id = context.node_config["partition-id"]
    model_cls = context.run_config["model"]
    dataset = context.run_config["dataset"]

    pmodel =  create_instantiate_parameters(dataset, model_cls)

    if personal_model_name not in context.state: 
        log(INFO, f"[Client {partition_id}] No Model found → training from scratch.")
        #Call the training function for personal 
        nn_model_training(pmodel, dataset, model_cls, context)
        save_model_from_to_state(context.state, pmodel)
    else:
        load_model_from_state(context.state, pmodel)
    
    # Construct and return reply Message
    model_record = ArrayRecord(pmodel.state_dict())
    model_meta_info = {
        "client_id": partition_id,
    }
    model_meta_info_record = MetricRecord(model_meta_info)
    content = RecordDict({"arrays": model_record, "meta_info": model_meta_info_record})
    return Message(content=content, reply_to=msg)

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    seed_all(context.run_config["seed"])
    strategy = context.run_config["strategy"]
    
    handler = TRAIN_HANDLERS.get(strategy)
    if handler is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(TRAIN_HANDLERS.keys())}")
    
    model_weights, metric_dict = handler(msg,context)

    # Construct and return reply Message
    model_record = ArrayRecord(model_weights)
    metric_record = MetricRecord(metric_dict)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    seed_all(context.run_config["seed"])
    # Load the data
    dataset = context.run_config["dataset"]

    # Load the model and initialize it with the received weights
    model_cls = context.run_config["model"]
    pmodel =  create_instantiate_parameters(dataset, model_cls)
    load_model_from_state(context.state, pmodel)

    metrics = nn_eval(pmodel, context)
    
    # Construct and return reply Message
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)