"""Linear noise schedule."""

import gin
import torch
from syncogen.diffusion.noise.base import NoiseBase


@gin.configurable
class LinearNoise(NoiseBase):
    """Linear noise schedule."""

    def __init__(
        self, sigma_min: float = 0, sigma_max: float = 10, dtype=torch.float32
    ):
        super().__init__()
        self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
        self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

    def rate_noise(self, t):
        """Rate of change of noise."""
        return self.sigma_max - self.sigma_min

    def total_noise(self, t):
        """Total noise."""
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def importance_sampling_transformation(self, t):
        """Transform timesteps for importance sampling."""
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        return (sigma_t - self.sigma_min) / (self.sigma_max - self.sigma_min)
