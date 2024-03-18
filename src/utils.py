import torch
import numpy as np
import random

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
