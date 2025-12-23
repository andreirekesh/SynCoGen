"""Cosine noise schedules."""

import gin
import torch
from syncogen.diffusion.noise.base import NoiseBase


@gin.configurable
class CosineNoise(NoiseBase):
    """Cosine noise schedule."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        """Rate of change of noise."""
        cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        """Total noise."""
        cos = torch.cos(t * torch.pi / 2)
        return -torch.log(self.eps + (1 - self.eps) * cos)


@gin.configurable
class CosineSqrNoise(NoiseBase):
    """Squared cosine noise schedule."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        """Rate of change of noise."""
        cos = (1 - self.eps) * (torch.cos(t * torch.pi / 2) ** 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        """Total noise."""
        cos = torch.cos(t * torch.pi / 2) ** 2
        return -torch.log(self.eps + (1 - self.eps) * cos)
