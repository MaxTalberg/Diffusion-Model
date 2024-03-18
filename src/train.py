import yaml
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from ddpm import DDPM
from cnn_model import CNN
from utils import set_seed
from utils import get_activation
from config_loader import load_config
from data_loader import get_dataloaders

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    train_losses = []
    pbar = tqdm(dataloader, desc='Training')

    for x, _ in pbar:
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_description(f"Train Loss: {np.mean(train_losses[-100:]):.3g}")
    
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss

def val_epoch(model, dataloader, device):
    model.eval()
    val_losses = []

    with torch.no_grad():
        for x, _ in dataloader:
            loss = model(x)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)
    return avg_val_loss

def train(config_path):

    # load config
    config = load_config(config_path)
    set_seed(config['hyperparameters']['seed'])

    # initialise the model
    cnn_config = config['CNN']
    cnn_config['act'] = get_activation(cnn_config['act'])
    ddpm_config = config['ddpm']
    ddpm_config['betas'] = [float(beta) for beta in ddpm_config['betas']]

    gt = CNN(**cnn_config)
    ddpm = DDPM(gt = gt, **config["ddpm"])
    optim = torch.optim.Adam(ddpm.parameters(), lr=float(config["optim"]["lr"]))

    # Load the MNIST dataset
    train_dataloader, test_dataloader = get_dataloaders(config["hyperparameters"]["batch_size"], 
                                      config["hyperparameters"]["num_workers"])

    # prepare the device
    accelerator = Accelerator()

    # We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
    # which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(ddpm, optim, train_dataloader, test_dataloader)
    
    # Train the model
    epochs = config['hyperparameters']['epochs']
    
    for epoch in range(epochs):
        avg_train_loss = train_epoch(ddpm, train_dataloader, optim, accelerator.device)
        avg_val_loss = val_epoch(ddpm, test_dataloader, accelerator.device)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}, Val Loss: {avg_val_loss:.3g}")
   
        # generate samples
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, f"./contents/ddpm_sample_{epoch:04d}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

if __name__ == "__main__":
    config = load_config("config.yaml")
    train("config.yaml")
