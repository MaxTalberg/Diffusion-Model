import torch
import numpy as np
from  tqdm import tqdm


from utils import save_training_results, frechet_distance
from plot_utils import save_and_plot_samples, plot_loss, plot_fid


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


def train(config, ddpm, optim, train_dataloader, accelerator, real_images, quick_test=False):

    # Initialize metrics
    metrics = []
    avg_train_losses = []
    fids = []

    # Training loop
    for epoch in range(config['hyperparameters']['epochs']):
        avg_train_loss = train_epoch(ddpm, train_dataloader, optim, single_batch=quick_test)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.3g}")

        # Generate samples and evaluate
        with torch.no_grad():
            xh, progress = ddpm.sample(16, (1, 28, 28), accelerator.device, timesteps=config["hyperparameters"]["timesteps"])
            save_and_plot_samples(xh, progress, epoch, ddpm, config["hyperparameters"]["timesteps"])

            if epoch % config["hyperparameters"]["interval"] == 0:
                fid_score = frechet_distance(real_images, xh)  # Ensure `real_images` are available
                fids.append(fid_score)
                print(f"FID Score: {fid_score}")

        # Update metrics
        epoch_metrics = {"epoch": epoch, "train_loss": avg_train_loss}
        metrics.append(epoch_metrics)

    # Save and plot final results
    save_training_results(config, metrics)
    plot_loss(avg_train_losses)
    plot_fid(fids, config["hyperparameters"]["interval"], config['hyperparameters']['epochs'])
