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
        grid_filename = os.path.join(save_path, f"progress_epoch{epoch:04d}.png")
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

    
    

'''def plot_progress(progress, timesteps):
    print("Progress content before sorting:")
    for idx, (t, img) in enumerate(progress):
        print(f"Index: {idx}, Timestep: {t}, Image tensor shape: {img.shape}")
    # Sort the progress list based on timesteps
    sorted_progress = sorted(progress, key=lambda x: timesteps.index(x[0]))
    
    # Create a list of images from the progress list
    images = [img_tensor[0] for _, img_tensor in progress]  # Extract only the first image from each mini-batch

    # Then create the grid
    grid = torchvision.utils.make_grid(images, nrow=len(images))
    # Convert grid to a numpy array
    np_grid = grid.cpu().numpy()
    
    # Transpose numpy array to (height, width, channels) from (channels, height, width)
    np_grid_transposed = np.transpose(np_grid, (1, 2, 0))
    
    # Plot the images
    plt.figure(figsize=(len(images) * 2, 2))
    plt.imshow(np_grid_transposed, interpolation='nearest')
    plt.axis('off')
    
    # Set titles for subplots
    for i, (t, _) in enumerate(sorted_progress):
        plt.text(i * images[0].shape[2], -10, f"t={t}", ha='center')
    
    plt.show()


def save_and_plot_samples(xh, progress, epoch, model, timesteps, save_path='./contents', model_path='./metrics', nrow=4):
    """
    Plots saved grids of generated images at specified epoch intervals.

    Parameters
    ----------
    epoch_interval : int, optional
        Interval between epochs to plot, by default 10.
    max_epoch : int, optional
        The maximum epoch to plot. If None, defaults to the highest epoch found, by default None.
    save_dir : str, optional
        The directory from which to load the saved grids, by default "./contents".

    Notes
    -----
    This function loads and plots saved grids of generated images from the specified directory. Each subplot is titled
    with the corresponding epoch. The function is useful for visualizing the progression of generated images over
    training epochs.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the generated sample grid
    grid = make_grid(xh, nrow=nrow)
    sample_filename = f"{save_path}/ddpm_sample_epoch_{epoch:04d}.png"
    save_image(grid, sample_filename)

    # Save the model state
    model_filename = os.path.join(model_path, "ddpm_mnist.pth")
    torch.save(model.state_dict(), model_filename)

    # Plot and save progress images with titles for each timestep
    # Extract the first image from each tensor in progress to represent each timestep
    representative_images = [tensor[0] for tensor in progress]  # List of first images from each tensor
    fig, axes = plt.subplots(1, len(representative_images), figsize=(2 * len(representative_images), 2))
    for ax, img, timestep in zip(axes, representative_images, timesteps):
        np_img = img.cpu().numpy().squeeze()
        ax.imshow(np_img, cmap='gray')
        adjusted_timestep = int(timestep) # Adjust timestep if necessary
        ax.set_title(fr"$Z_{{{adjusted_timestep}}}$", fontsize=10)
        ax.axis('off')

    progress_filename = f"{save_path}/progress_epoch_{epoch:04d}.png"
    plt.savefig(progress_filename, bbox_inches='tight')
    plt.close(fig)
'''
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
