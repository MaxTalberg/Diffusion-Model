import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from PIL import Image



def save_and_plot_samples(xh, progress, epoch, model, timesteps, save_path='./contents', model_path='./', nrow=4):
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
        adjusted_timestep = 1000 - timestep  # Adjust timestep if necessary
        ax.set_title(fr"$Z_{{{adjusted_timestep}}}$", fontsize=10)
        ax.axis('off')

    progress_filename = f"{save_path}/progress_epoch_{epoch:04d}.png"
    plt.savefig(progress_filename, bbox_inches='tight')
    plt.close(fig)


def plot_saved_grids(epoch_interval=10, max_epoch=None, save_dir="./contents"):

    images = []
    titles = []

    # Determine the range of epochs to include
    epochs = range(0, max_epoch + 1, epoch_interval) if max_epoch is not None else range(0, 1001, epoch_interval)

    for epoch in epochs:
        # Construct the filename
        filename = f"ddpm_sample_{epoch:04d}.png"
        filepath = os.path.join(save_dir, filename)
        # Load the image if it exists
        if os.path.exists(filepath):
            img = Image.open(filepath)
            images.append(img)
            titles.append(f"Epoch = {epoch+1}")


    # Create a single figure with multiple subplots
    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2)) 
    
    # if only one image
    if len(images) == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.savefig(f"./contents/ddpm_sample_{int(time.time())}.png")
    plt.close(fig)

def plot_loss(avg_train_losses):

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses, label='Average Train Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_fid(fids, interval, epochs):
    plt.figure(figsize=(10, 5))
    epochs_plotted = list(range(0, epochs, interval))
    plt.plot(epochs_plotted, fids)
    plt.xlabel('Epoch')
    plt.ylabel('Frechet inception distance')
    plt.xticks(epochs_plotted) 
    plt.show()
