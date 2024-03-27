import torch
from typing import Dict


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Computes the noise schedules for Denoising Diffusion Probabilistic Models (DDPM)
    based on a linear progression of noise levels defined by
    `beta1` and `beta2` over `T` timesteps. This function returns a dictionary
    containing the tensors `beta_t` and `alpha_t`, which represent the noise schedule
    and the corresponding cumulative product of (1 - beta), respectively.

    Parameters
    ----------
    beta1 : float
        The starting noise level, representing the lower bound of the linear noise
        schedule. Must be greater than 0 and less than `beta2`.
    beta2 : float
        The ending noise level, representing the upper bound of the linear noise
        schedule. Must be greater than `beta1` and less than 1.
    T : int
        The total number of timesteps in the diffusion process.
        Determines the length of the noise schedule.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing the following keys:
        - 'beta_t': A tensor of shape (T+1,) representing the linearly spaced noise
        levels from `beta1` to `beta2`.
        - 'alpha_t': A tensor of shape (T+1,) representing the cumulative product of
        (1 - beta_t) values, computed in log-space for numerical stability.

    Notes
    -----
    - The values of `beta1` and `beta2` must satisfy the condition 0 < beta1 < beta2 < 1
    to ensure a valid noise schedule.
    - The computed `alpha_t` values are used to scale the data during the forward and
    reverse diffusion processes to maintain stability and ensure the model converges
    to the data distribution.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}
