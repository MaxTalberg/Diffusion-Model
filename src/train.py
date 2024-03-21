import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.utils import save_image, make_grid

from ddpm import DDPM
from cnn_model import CNN
from utils import load_config, set_seed, frechet_distance
from plot_utils import plot_loss, plot_saved_grids, plot_fid, save_and_plot_samples
from data_loader import get_dataloaders
from utils import save_training_results


def train_epoch(model, dataloader, optimizer, single_batch=False):
    model.train()
    train_losses = []

    pbar = tqdm(dataloader, desc='Training', total=(1 if single_batch else None))

    for x, _ in pbar:
        optimizer.zero_grad()
        loss, zt = model.forward(x)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_description(f"Train Loss: {np.mean(train_losses):.3g}")
        
        if single_batch:
            break  # Exit after first batch if single_batch is True

    # Calculate average loss (per epoch)
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss


def train(config_path, quick_test=False):

    # store metrics
    metrics = []
    avg_train_losses = []
    fids = []
    epoch_metrics = {}

    # load config
    config = load_config(config_path)

    # set seed
    set_seed(config['hyperparameters']['seed'])

    # initialise the model
    gt = CNN(**config['CNN'])
    ddpm = DDPM(gt = gt, **config["ddpm"])
    optim = torch.optim.Adam(ddpm.parameters(), lr=float(config["optim"]["lr"]))
    timesteps = config["hyperparameters"]["timesteps"]
    interval = config["hyperparameters"]["interval"]

    # Load the MNIST dataset
    train_dataloader, test_dataloader = get_dataloaders(config["hyperparameters"]["batch_size"], 
                                      config["hyperparameters"]["num_workers"])
    
    # get real images for FID
    real_images, _ = next(iter(train_dataloader))

    # prepare the device
    accelerator = Accelerator()
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(ddpm, optim, train_dataloader, test_dataloader)
    
    # Train the model
    epochs = config['hyperparameters']['epochs']
    
    for epoch in range(epochs):

        # training loss
        avg_train_loss = train_epoch(ddpm, train_dataloader, optim, single_batch=quick_test)

        # generate samples
        with torch.no_grad():
            xh, progress = ddpm.sample(16, (1, 28, 28), accelerator.device, timesteps=timesteps)
            save_and_plot_samples(xh, progress, epoch, ddpm, timesteps)
        
            if epoch % interval == 0:
                fid_score = frechet_distance(real_images, xh)
                fid_score = float(fid_score)
                fids.append(fid_score)
                epoch_metrics["fid_score"] = fid_score
                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}FID Score {fid_score}")
            else:
                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}")


        # Append epoch and metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss
        }

        metrics.append(epoch_metrics)
        avg_train_losses.append(avg_train_loss)

    # save metrics
    save_training_results(config, metrics)

    # plots
    plot_saved_grids(epoch_interval=interval, max_epoch=epochs+1)
    plot_loss(avg_train_losses)
    plot_fid(fids, interval, epochs)

if __name__ == "__main__":
    config = load_config("config.yaml")
    train("config.yaml", quick_test=False)  # Set to False for full training
