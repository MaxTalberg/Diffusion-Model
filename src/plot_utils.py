import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

import torchvision
import numpy as np

from PIL import Image

def plot_progress(xh, progress,  epoch, timesteps, cold_diff = False, save_path='./contents'):
    os.makedirs(save_path, exist_ok=True)

    # Save the generated sample grid
    grid = make_grid(xh, nrow=4)
    sample_filename = f"{save_path}/ddpm_sample_epoch_{epoch:04d}.png"
    save_image(grid, sample_filename)

    
    # Sort the progress list based on timesteps
    sorted_progress = sorted(progress, key=lambda x: timesteps.index(x[0]))
    
    # Extract the first image from each mini-batch to represent each timestep
    images = [img_tensor[0] for _, img_tensor in sorted_progress]

    if cold_diff:
            # If you also want to save the grid image like the second function:
        grid = make_grid(images, nrow=len(images))
        grid_filename = os.path.join(save_path, f"progress_epoch_{epoch:04d}.png")
        save_image(grid, grid_filename)
    
    else:

        fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2))
        for ax, img, timestep in zip(axes, images, timesteps):
            np_img = img.cpu().numpy().squeeze()
            ax.imshow(np_img, cmap='gray')
            adjusted_timestep = int(timestep) # Adjust timestep if necessary
            ax.set_title(fr"$Z_{{{adjusted_timestep}}}$", fontsize=10)
            ax.axis('off')

        progress_filename = f"{save_path}/progress_epoch_{epoch:04d}.png"
        plt.savefig(progress_filename, bbox_inches='tight')
        plt.close(fig)

    
def plot_loss(avg_train_losses):
    """
    Plots the average training loss per epoch.

    Parameters
    ----------
    avg_train_losses : list of float
        A list containing the average training loss for each epoch.

    Notes
    -----
    This function plots the average training loss over epochs, providing insight into how the model's
    training is progressing in terms of minimizing the loss over time.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses, label='Average Train Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_fid(fids, interval, epochs):
    """
    Plots the Frechet Inception Distance (FID) score over epochs.

    Parameters
    ----------
    fids : list of float
        A list containing the FID scores at specified intervals.
    interval : int
        The interval at which FID scores were calculated.
    epochs : int
        The total number of epochs.

    Notes
    -----
    This function plots the FID scores, which are calculated at specified intervals over the total number of epochs. 
    The FID score is a measure of similarity between two sets of images, and in this context, it usually measures how 
    similar generated images are to real images. Lower FID scores indicate better image quality and feature similarity.
    """
    
    plt.figure(figsize=(10, 5))
    epochs_plotted = list(range(0, epochs, interval))
    plt.plot(epochs_plotted, fids, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score over Epochs')
    plt.xticks(epochs_plotted)
    plt.grid(True)
    plt.show()

def plot_metrics(metrics, config, fid_score=False):
    """
    Plots the training loss and FID score from the training metrics.

    Parameters
    ----------
    metrics : list of dicts
        A list where each element is a dictionary containing 'epoch', 'train_loss', and optionally 'fid_score'.
    config : dict
        The configuration dictionary containing training parameters and hyperparameters.
    """
    # Extracting training losses and FID scores
    avg_train_losses = [m['train_loss'] for m in metrics]
    fids = [m['fid_score'] for m in metrics if 'fid_score' in m]
    epochs = config['hyperparameters']['epochs']
    interval = config['hyperparameters']['interval']

    plot_loss(avg_train_losses)
    if fid_score:
        plot_fid(fids, interval, epochs)
    




