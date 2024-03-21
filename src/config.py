import yaml
import torch
import random
import numpy as np

from cnn import CNN
from ddpm import DDPM
from accelerate import Accelerator
from data_loader import get_dataloaders

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_environment(config_path: str):
    # Load configuration
    config = load_config(config_path)

    # Set seed for reproducibility
    set_seed(config['hyperparameters']['seed'])

    # Initialize the model and optimizer
    gt = CNN(**config['CNN'])
    ddpm = DDPM(gt=gt, **config["ddpm"])
    optim = torch.optim.Adam(ddpm.parameters(), lr=float(config["optim"]["lr"]))

    # Load the dataset
    train_dataloader, test_dataloader = get_dataloaders(
        config["hyperparameters"]["batch_size"],
        config["hyperparameters"]["num_workers"]
    )

    # real images for fid
    real_images,_ = next(iter(train_dataloader))

    # Prepare the device (GPU/CPU)
    accelerator = Accelerator()
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(
        ddpm, optim, train_dataloader, test_dataloader
    )

    return config, ddpm, optim, train_dataloader, test_dataloader, accelerator, real_images
