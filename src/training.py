import torch
import numpy as np
from  tqdm import tqdm


from utils import save_training_results, frechet_distance
from plot_utils import save_and_plot_samples, plot_loss, plot_fid


def train_epoch(model, dataloader, optimizer, single_batch=False):
    """
    Performs a single epoch of training.

    Parameters
    ----------
    model : torch.nn.Module
        The model that is being trained.
    dataloader : torch.utils.data.DataLoader
        The DataLoader providing the training data.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters based on gradients.
    single_batch : bool, optional
        If True, the training will be performed on a single batch only. This is useful for quick testing. Default is False.

    Returns
    -------
    float
        The average training loss for the epoch.

    Notes
    -----
    This function iterates over the dataloader, performing forward passes, computing losses, and backpropagation
    for each batch. If `single_batch` is True, the loop exits after processing the first batch.
    """
    model.train()
    train_losses = []

    pbar = tqdm(dataloader, desc='Training', total=(1 if single_batch else None))

    for x, _ in pbar:
        optimizer.zero_grad()
        loss, _ = model.forward(x)  # Assume model.forward returns a tuple (loss, some_other_value)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_description(f"Train Loss: {np.mean(train_losses):.3g}")
        
        if single_batch:
            break

    avg_train_loss = np.mean(train_losses)
    return avg_train_loss


def train(config, ddpm, optim, train_dataloader, accelerator, real_images, quick_test=False):
    """
    Executes the training process.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing training parameters and hyperparameters.
    ddpm : DDPM
        The initialized DDPM (Denoising Diffusion Probabilistic Models) model to be trained.
    optim : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    train_dataloader : torch.utils.data.DataLoader
        The DataLoader providing the training data.
    accelerator : Accelerator
        The Hugging Face Accelerator for mixed precision and distributed training.
    real_images : torch.Tensor
        A batch of real images used for calculating the Frechet Inception Distance (FID).
    quick_test : bool, optional
        If True, performs training on a single batch only for quick testing. Default is False.

    Notes
    -----
    This function runs the training loop for a specified number of epochs defined in the configuration. At each epoch,
    it optionally evaluates the model by generating samples and calculating the FID score. Progress images are saved
    periodically, and training metrics are plotted at the end of training.
    """
    metrics = []
    avg_train_losses = []
    fids = []

    for epoch in range(config['hyperparameters']['epochs']):
        avg_train_loss = train_epoch(ddpm, train_dataloader, optim, single_batch=quick_test)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}")

        epoch_metrics = {"epoch": epoch, "train_loss": avg_train_loss}

        with torch.no_grad():
            xh, progress = ddpm.sample(16, (1, 28, 28), accelerator.device, timesteps=config["hyperparameters"]["timesteps"])
            save_and_plot_samples(xh, progress, epoch, ddpm, config["hyperparameters"]["timesteps"], config["ddpm"]["n_T"])

            if epoch % config["hyperparameters"]["interval"] == 0:
                fid_score = frechet_distance(real_images, xh)
                fid_score = float(fid_score)
                fids.append(fid_score)
                epoch_metrics["fid_score"] = fid_score
                print(f"FID Score: {fid_score}")

        metrics.append(epoch_metrics)

    save_training_results(config, metrics)
    plot_loss(avg_train_losses)
    #plot_fid(fids, config["hyperparameters"]["interval"], config['hyperparameters']['epochs'])
