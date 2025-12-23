"""Log-linear noise schedule."""

import gin
import torch
from syncogen.diffusion.noise.base import NoiseBase


@gin.configurable
class LogLinearNoise(NoiseBase):
    """Log-linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t varies from 0 to 1.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        """Rate of change of noise."""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        """Total noise."""
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        """Transform timesteps for importance sampling."""
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        t = -torch.expm1(-sigma_t) / (1 - self.eps)
        return t
