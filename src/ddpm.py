import torch
import torch.nn as nn
from ddpm_schedule import ddpm_schedules

from torchvision.transforms import GaussianBlur
from data_loader import get_dataloaders
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        beta1: float,
        beta2: float,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(float(beta1), float(beta2), n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model, simulating one step of the diffusion process as described in Algorithm 18.1 from Prince. It applies a transformation to the input tensor `x` to produce a noised version `z_t` and computes a loss based on the predicted noise.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the diffusion step, typically representing an image or batch of images.

        Returns
        -------
        tuple
            A tuple containing the computed loss and the noised version of the input tensor `z_t`.
        
        Notes
        -----
        Algorithm 18.1 in Prince
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T)), z_t

    def sample(self, n_sample: int, size, device, timesteps=None) -> torch.Tensor:
        """
        Generates samples by reversing the diffusion process as described in Algorithm 18.2 from Prince. This method iteratively refines noise to generate samples, optionally recording the intermediate states at specified timesteps.

        Parameters
        ----------
        n_sample : int
            The number of samples to generate.
        size : tuple
            The size of each sample to generate.
        device : torch.device
            The device on which to perform the computation.
        timesteps : list, optional
            Specific timesteps at which to record the intermediate states of the generated samples.

        Returns
        -------
        torch.Tensor, list
            The final generated samples and a list of tuples containing the timestep and the sample at that timestep, if `timesteps` is provided.

        Notes
        -----
        Algorithm 18.2 in Prince
        """
        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        progress = []

        for i in range(self.n_T, 0, -1):

            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)

            if i in timesteps:
                timestep = i.item() if isinstance(i, torch.Tensor) else i
                progress.append((timestep, z_t.clone()))

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t, progress

    def blurrer(self, t, item = True):
        """
        Creates a Gaussian blurring operation based on the specified timestep `t`. The blur intensity is determined by the timestep, with higher timesteps resulting in more blur.

        Parameters
        ----------
        t : torch.Tensor or int
            The timestep based on which the blur intensity is calculated. If `t` is a tensor, its item will be used if `item` is True.
        item : bool, default True
            Specifies whether to call the `.item()` method on `t` if it's a tensor to get a Python number.

        Returns
        -------
        torchvision.transforms.GaussianBlur
            A Gaussian blur transform configured with the calculated sigma value based on the timestep.
        """
        sigma_base = 0.02
        sigma_scale = 0.01
        if item:
            sigma_value = sigma_base + (sigma_scale * t.item())
        else:
            sigma_value = sigma_base + (sigma_scale * t)
        return GaussianBlur(kernel_size=(29,29), sigma=(sigma_value))
    
    def forward_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a forward pass with blurring to the input tensor `x`. This method simulates a blurring process on the input images and computes a loss based on the predicted original images from the blurred images.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing a batch of images to be blurred and processed.

        Returns
        -------
        tuple
            A tuple containing the computed loss and the blurred version of the input tensor.
        """
        # random timestep
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        
        # hold blurred images
        blurred_images = []

        # initialise blurrer
        for i in range(x.shape[0]):
            blurrer = self.blurrer(t[i])
            blurred_img = blurrer(x[i].unsqueeze(0))
            blurred_images.append(blurred_img.squeeze(0))

        blurred_z_t = torch.stack(blurred_images)
   
        return self.criterion(x, self.gt(blurred_z_t, t / self.n_T)), blurred_z_t


    def sample_blur(self, n_sample: int, device, timesteps=None) -> torch.Tensor:
        """
        Generates blurred samples by applying a specific blurring process described in Algorithm 2 from Bansal et al. (2020). This method simulates the process of unblurring the images step by step, optionally recording the intermediate states at specified timesteps.

        Parameters
        ----------
        n_sample : int
            The number of samples to generate.
        device : torch.device
            The device on which to perform the computation.
        timesteps : list, optional
            Specific timesteps at which to record the intermediate states of the generated samples.

        Returns
        -------
        torch.Tensor, list
            The final generated (unblurred) samples and a list of tuples containing the timestep and the sample at that timestep, if `timesteps` is provided.
        
        Notes
        -----
        Algorithm 2 in Bansal et al. (2020)
        """

        # max z_t
        train_dataloader = get_dataloaders(16, 8)

        # get images
        x, _ = next(iter(train_dataloader))

        # blur images
        blurrer = self.blurrer(self.n_T, item=False)
        z_t = blurrer(x).float().to(device)
        
        # initialise Z_T
        _one = torch.ones(n_sample, device=device)
        progress = []

        for i in range(self.n_T, 0, -1):

            x0_pred = self.gt(z_t, (i/self.n_T) * _one)

            if i in timesteps:
                timestep = i.item() if isinstance(i, torch.Tensor) else i
                progress.append((timestep, z_t.clone()))

            # initialise blurrer
            blurrer_t = self.blurrer(i, item=False)
            blurrer_tm1 = self.blurrer(i-1, item=False)
            z_t = (z_t 
                   - blurrer_t(x0_pred) 
                   + blurrer_tm1(x0_pred))
            
        return z_t, progress
