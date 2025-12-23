"""Geometric noise schedule."""

import gin
import torch
from syncogen.diffusion.noise.base import NoiseBase


@gin.configurable
class GeometricNoise(NoiseBase):
    """Geometric noise schedule."""

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t):
        """Rate of change of noise."""
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(self, t):
        """Total noise."""
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t
