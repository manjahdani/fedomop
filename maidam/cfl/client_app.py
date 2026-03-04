import os
from flwr.client import ClientApp
from flwr.common import Context
from logging import Formatter, FileHandler
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from logging import INFO, WARNING
from flwr.common.logger import log
from collections.abc import Iterable
from typing import Callable, Optional, Tuple
from maidam.cfl.clients_strategy.fed_avg_client import train_fedavg, eval_fedavg
from maidam.cfl.clients_strategy.fed_per_client import train_fedper, eval_fedper
from maidam.cfl.clients_strategy.ditto_client import train_ditto, eval_ditto
from maidam.ml.fl_common.task_utils import seed_all

# Flower ClientApp
app = ClientApp()

TrainHandler = Callable[[Message, Context], Tuple[ArrayRecord, MetricRecord]]
EvaluateHandler =  Callable[[Message, Context], Tuple[ArrayRecord, MetricRecord]]


TRAIN_HANDLERS: dict[str, TrainHandler] = {
    "FedAvg": train_fedavg,
    "FedPer": train_fedper,
    "Ditto" : train_ditto,
    # "cfl": train_cfl,
}

EVALUATE_HANDLERS: dict[str, TrainHandler] = {
    "FedAvg": eval_fedavg,
    "FedPer": eval_fedper,
    "Ditto" : eval_ditto,
    # "cfl": train_cfl,
}

@app.train()
def train(msg: Message, context: Context):
    
    seed_all(context.run_config["seed"])
    
    strategy = context.run_config["strategy"]
    handler = TRAIN_HANDLERS.get(strategy)
    if handler is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(TRAIN_HANDLERS.keys())}")

    model_weights, metric_dict = handler(msg, context)

    # Construct and return reply Message
    model_record = ArrayRecord(model_weights)
    metric_record = MetricRecord(metric_dict)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    #print(f"id {context.node_config["partition-id"]}")
    return Message(content=content, reply_to=msg)



@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    seed_all(context.run_config["seed"])
    
    strategy = context.run_config["strategy"]
    handler = EVALUATE_HANDLERS.get(strategy)
    if handler is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(EVALUATE_HANDLERS.keys())}")

    metrics = handler(msg,context)
    
    metric_record = MetricRecord(metrics)


    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)