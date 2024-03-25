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
        """Algorithm 18.1 in Prince"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T)), z_t

    def sample(self, n_sample: int, size, device, timesteps=None) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

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
                progress.append(z_t.clone())

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t, progress

    def blurrer(self, t, item = True):
        sigma_base = 0.02
        sigma_scale = 0.05
        if item:
            sigma_value = sigma_base + (sigma_scale * t.item())
        else:
            sigma_value = sigma_base + (sigma_scale * t)
        return GaussianBlur(kernel_size=(29,29), sigma=(sigma_value))
    
    def forward_blur(self, x: torch.Tensor) -> torch.Tensor:

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
    '''    
    def max_blur(self, x: torch.Tensor) -> torch.Tensor:

        sigma_base = 0.2
        sigma_scale = 0.2
        sigma_value = sigma_base + sigma_scale * math.log(self.n_T + 1)
        blurrer = GaussianBlur(kernel_size=(5,9), sigma=(sigma_value))

        return blurrer(x)'''

    def sample_blur(self, n_sample: int, size, device, timesteps=None) -> torch.Tensor:

        # max z_t
        train_dataloader, _ = get_dataloaders(16, 8)
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
