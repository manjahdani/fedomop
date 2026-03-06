import json
import os
from pathlib import Path
import time

from flwr.serverapp.strategy import Result

RESULTS_FILE = "result-{}.json"

def config_json_file(n_nodes : int , run_config: dict) -> None:
    """Initialize the json file and write the run configurations."""
    # Initialize the execution results directory.
    res_save_path = f"./results/{run_config['dataset']}/{str(n_nodes)}_clients/{run_config['num-server-rounds']}_rounds"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)
    res_save_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Set the date and full path of the file to save the results.
    global RESULTS_FILE 
    RESULTS_FILE = RESULTS_FILE.format(res_save_name)
    RESULTS_FILE = f"{res_save_path}/{RESULTS_FILE}"

    data = {
        "number_of_nodes": n_nodes,
        "run_config": dict(run_config.items()),
        "round_res": [],
    }
    with open(RESULTS_FILE, "w+", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)
    return Path(RESULTS_FILE)

def save_metrics_as_json(save_path: str, result: Result) -> None:
    """Append per-round metrics into payload['round_res'] in save_path (JSON file)."""

    # save_path is the JSON file path
    with open(save_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)  # dict with keys: run_config, round_res

    # Ensure round_res exists and is a list
    round_res = payload.setdefault("round_res", [])
    round_ids = sorted(set(result.evaluate_metrics_clientapp.keys()))
    
    for r in round_ids:
        round_res.append({
            "round": r,
            "evaluate_metrics_clientapp": dict(result.evaluate_metrics_clientapp.get(r, {})),
        })

    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)