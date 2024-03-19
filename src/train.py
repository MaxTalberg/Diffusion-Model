import os
import torch
import random
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image, make_grid

from ddpm import DDPM
from cnn_model import CNN
from utils import load_config, set_seed, frechet_distance
from plot_utils import plot_loss, plot_progress, plot_saved_grids, plot_fid
from data_loader import get_dataloaders
from utils import save_training_results


def train_epoch(model, dataloader, optimizer, single_batch=False):
    model.train()
    train_losses = []
    pbar = tqdm(dataloader, desc='Training', total=(1 if single_batch else None))

    for x, _ in pbar:
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_description(f"Train Loss: {np.mean(train_losses):.3g}")
        
        if single_batch:
            break  # Exit after first batch if single_batch is True
    
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss

def val_epoch(model, dataloader, single_batch=False):
    model.eval()
    val_losses = []

    with torch.no_grad():
        for x, _ in dataloader:
            loss = model(x)
            val_losses.append(loss.item())

            if single_batch:
                break

    avg_val_loss = np.mean(val_losses)
    return avg_val_loss

def train(config_path, quick_test=False):

    # store metrics
    metrics = []
    avg_train_losses = []
    avg_val_losses = []
    fids = []

    # load config
    config = load_config(config_path)

    # set seed
    set_seed(config['hyperparameters']['seed'])

    # initialise the model
    gt = CNN(**config['CNN'])
    ddpm = DDPM(gt = gt, **config["ddpm"])
    optim = torch.optim.Adam(ddpm.parameters(), lr=float(config["optim"]["lr"]))
    timesteps = config["hyperparameters"]["timesteps"]

    # Load the MNIST dataset
    train_dataloader, test_dataloader = get_dataloaders(config["hyperparameters"]["batch_size"], 
                                      config["hyperparameters"]["num_workers"])
    
    # get real images for FID
    real_images, _ = next(iter(test_dataloader))

    # prepare the device
    accelerator = Accelerator()
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(ddpm, optim, train_dataloader, test_dataloader)
    
    # Train the model
    epochs = config['hyperparameters']['epochs']
    
    for epoch in range(epochs):
        avg_train_loss = train_epoch(ddpm, train_dataloader, optim, single_batch=quick_test)
        avg_val_loss = val_epoch(ddpm, test_dataloader, single_batch=quick_test)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}, Val Loss: {avg_val_loss:.3g}")

        # Append epoch and metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }
        metrics.append(epoch_metrics)
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)

        # generate samples
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)
            # Save samples to `./contents` directory
            save_image(grid, f"./contents/ddpm_sample_{epoch:04d}.png")
            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")
            progress_images = ddpm.sample_with_progress(1, (1, 28, 28), accelerator.device, timesteps=timesteps)
            plot_progress(progress_images, timesteps, epoch)
            # run every other
            if (epoch+1) % 10 == 0:
                fids.append(frechet_distance(real_images, xh))
    
    # save metrics
    save_training_results(config, metrics)

    # plot saved grids
    plot_saved_grids(epoch_interval=2, max_epoch=11)

    # plot loss
    plot_loss(avg_train_losses, avg_val_losses)

    # plot FID
    #plot_fid(fids, epochs=epochs)

if __name__ == "__main__":
    config = load_config("config.yaml")
    train("config.yaml", quick_test=True)  # Set to False for full training
