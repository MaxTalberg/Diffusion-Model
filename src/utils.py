import torch
import numpy as np
import random
import json
import time
import os


def get_activation(name):
    if name == "nn.GELU":
        return torch.nn.GELU
    elif name == "nn.ReLU":
        return torch.nn.ReLU
    else:
        raise ValueError(f"Activation {name} not found")

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