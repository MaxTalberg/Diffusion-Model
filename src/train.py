import yaml
import torch
import random
import numpy as np
from torch import nn
from ddpm import DDPM
from tqdm import tqdm
from cnn_model import CNN
from accelerate import Accelerator
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

# Load the configuration file
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# conert activitation function name to class
def get_activation(name):
    if name == "nn.GELU":
        return nn.GELU
    elif name == "nn.ReLU":
        return nn.ReLU
    else:
        raise ValueError(f"Activation {name} not found")

def train(config_path):

    # load config
    config = load_config(config_path)

    # Initialise the random seed
    seed = config['hyperparameters']['seed']  # Assuming seed is under hyperparameters
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(config['seed'])

    # extract config
    cnn_config = config['CNN']
    cnn_config['act'] = get_activation(cnn_config['act'])
    ddpm_config = config['ddpm']
    ddpm_config['betas'] = [float(beta) for beta in ddpm_config['betas']]

    gt = CNN(**cnn_config)
    ddpm = DDPM(gt = gt, **config["ddpm"])
    optim = torch.optim.Adam(ddpm.parameters(), lr=float(config["optim"]["lr"]))

    # Load the MNIST dataset
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    train_dataset = MNIST("./data", train=True, download=True, transform=tf)
    test_dataset = MNIST("./data", train=False, download=True, transform=tf)
    train_dataloader = DataLoader(train_dataset, batch_size=config["hyperparameters"]["batch_size"], 
                                  shuffle=True, num_workers=config["hyperparameters"]["num_workers"], drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["hyperparameters"]["batch_size"], 
                                 shuffle=False, num_workers=config["hyperparameters"]["num_workers"], drop_last=True)

    # prepare the device
    accelerator = Accelerator()

    # We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
    # which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(ddpm, optim, train_dataloader, test_dataloader)
    
    # Train the model
    epochs = config['hyperparameters']['epochs']
    train_losses = []
    val_losses = []
    avg_train_losses_per_epoch = []
    avg_val_losses_per_epoch = []

    for epoch in range(epochs):
        # training
        ddpm.train()
        pbar = tqdm(train_dataloader)
        temp_train_losses = []

        for x, _ in pbar:
            optim.zero_grad()
            train_loss = ddpm(x)
            train_loss.backward()
            optim.step()

            temp_train_losses.append(train_loss.item())
            pbar.set_description(f"Epoch {epoch} - Train Loss: {np.mean(temp_train_losses[-100:]):.3g}")

        
        avg_train_loss = np.mean(temp_train_losses)
        avg_train_losses_per_epoch.append(avg_train_loss)
            
        # validation
        ddpm.eval()

        temp_val_losses = []
        with torch.no_grad():
            for x, _ in test_dataloader:
                val_loss = ddpm(x)
                temp_val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(temp_val_losses)
            avg_val_losses_per_epoch.append(avg_val_loss)
            print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.3g}")
            
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
