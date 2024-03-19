import torch
import numpy as np
import random
import json
import time
import os
import yaml

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_training_results(config, metrics):
    results = {
        "config": config,
        "metrics": metrics
    }

    path = 'metrics/'
    filename = f"results_{int(time.time())}.json"

    with open(os.path.join(path, filename), "w") as outfile:
        json.dump(results, outfile, indent=4)