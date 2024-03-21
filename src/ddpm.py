import torch
import torch.nn as nn
from ddpm_schedule import ddpm_schedules

from PIL import Image
from torchvision.transforms import v2


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
            if i in timesteps:
                progress.append(z_t.clone())

            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t, progress


    def forward_blur(self, x: torch.Tensor) -> torch.Tensor:

        # random timestep
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        # initialise blurrer
        blurrer = v2.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 2.))

        # blur error term
        blurred_z_t = blurrer(x)
        blurred_eps = blurred_z_t - x

                
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting


        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * blurred_eps


        return self.criterion(self.gt(x, t / self.n_T), z_t), z_t

    def max_blur(self, x: torch.Tensor) -> torch.Tensor:

        # initialise blurrer
        blurrer = v2.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 2.))

        # blur error term
        blurred_z_t = blurrer(x)
        blurred_eps = blurred_z_t - x

        alpha_t = self.alpha_t[self.n_T, None, None, None]

        z_T = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * blurred_eps

        return z_T
    

    def sample_blur(self, x, n_sample: int, device) -> torch.Tensor:


        # initialise Z_T
        _one = torch.ones(n_sample, device=device)
        #z_t = torch.zeros(n_sample, *size, device=device)
        z_t = self.max_blur(x)

        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            z_t0 = z_t.clone()

            # deblurring step
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)


            if i > 1:
                # correction step
                correction_step = (torch.sqrt(beta_t) * torch.randn_like(z_t)) - (torch.sqrt(beta_t) * torch.randn_like(z_t0))
                z_t += correction_step

        return z_t
    
 
    def sample_blur_progress(self, x, n_sample: int, device, timesteps=None) -> torch.Tensor:

        if timesteps is None:
            timesteps = [self.n_T]  # Default to only the final step
        
        _one = torch.ones(n_sample, device=device)
        z_t = self.max_blur(x)
        progress_images = []
        
        for i in range(self.n_T, 0, -1):
            if i in timesteps:
                progress_images.append(z_t.clone())  # Clone to avoid in-place modifications
            
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            z_t0 = z_t.clone()

            # deblurring step
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)


            if i > 1:
                # correction step
                correction_step = (torch.sqrt(beta_t) * torch.randn_like(z_t)) - (torch.sqrt(beta_t) * torch.randn_like(z_t0))
                z_t += correction_step

        return progress_images
    

        
    '''def max_blur(self, x: torch.Tensor) -> torch.Tensor:

        # initialise blurrer
        blurrer = v2.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 2.))

        # blur error term
        blurred_z_t = blurrer(x)
        blurred_eps = blurred_z_t - x

        alpha_t = self.alpha_t[self.n_T, None, None, None]

        z_T = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * blurred_eps

        return z_T'''