from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import torch

from fedomop.task_utils import seed_all

from fedomop.clients_strategy.fed_per import train_fedper, eval_fedper
from fedomop.clients_strategy.self_training import train_self, eval_self
from fedomop.clients_strategy.fed_avg import train_fedavg, eval_fedavg

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    seed_all(context.run_config["seed"])
    
    if context.run_config["strategy"] == "FedAvg":
        model_weights, metric_dict = train_fedavg(msg, context)
    elif context.run_config["strategy"] == "FedPer":
        model_weights, metric_dict = train_fedper(msg, context)
    elif context.run_config["strategy"] == "Self":
        model_weights, metric_dict = train_self(msg, context)

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
    
    if context.run_config["strategy"] == "FedAvg":
        metrics = eval_fedavg(msg, context)
    elif context.run_config["strategy"] == "FedPer":
        metrics = eval_fedper(msg, context)
    elif context.run_config["strategy"] == "Self":
        metrics = eval_self(msg, context)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)