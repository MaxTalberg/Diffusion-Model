import yaml
import torch
import random
import numpy as np

from cnn import CNN
from ddpm import DDPM
from accelerate import Accelerator
from data_loader import get_dataloaders


def set_seed(seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use for random number generators in numpy, random, and torch.

    Notes
    -----
    This function sets the seed for numpy, random, and torch random number generators, including
    CUDA's random number generator if CUDA is available. This is essential for reproducibility
    of results in experiments involving random number generation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Parameters
    ----------
    config_path : str
        The file path to the configuration file.

    Returns
    -------
    dict
        The configuration loaded from the YAML file.

    Notes
    -----
    This function uses `yaml.safe_load` to load the configuration, which is considered safer
    than `yaml.load` without specifying a Loader.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_environment(config_path: str):
    """
    Sets up the training environment by loading the configuration, initializing the model and
    optimizer, loading the dataset, and preparing the device.

    Parameters
    ----------
    config_path : str
        The file path to the configuration file.

    Returns
    -------
    tuple
        A tuple containing the loaded configuration, initialized model, optimizer, training and
        testing dataloaders, accelerator object, and a batch of real images for FID calculation.

    Notes
    -----
    This function is a high-level setup function that prepares everything needed to start training.
    It initializes the CNN and DDPM models based on the provided configuration, sets up the dataloaders,
    and prepares the device for training using the Hugging Face Accelerator for mixed precision and
    distributed training. It also extracts a batch of real images from the training dataloader for
    later use in FID score calculation.
    """
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

    # Extract a batch of real images for FID
    real_images, _ = next(iter(train_dataloader))

    # Prepare the device (GPU/CPU)
    accelerator = Accelerator()
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(
        ddpm, optim, train_dataloader, test_dataloader
    )

    return config, ddpm, optim, train_dataloader, test_dataloader, accelerator, real_images