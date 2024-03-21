import os
import time
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image

def plot_progress(images, timesteps, epoch, nrow=4):
    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2))
    for ax, img, timestep in zip(axes, images, timesteps):
        ax.imshow(img.cpu().numpy().squeeze(), cmap='gray')
        adjusted_timestep = 1000 - timestep
        ax.set_title(fr"$Z_{{{adjusted_timestep}}}$", fontsize=10)
        ax.axis('off')

    plt.savefig(f"./contents/ddpm_progress_{epoch:04d}.png")
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
