"""Brownian bridge noise schedule."""

import gin
import torch
from syncogen.diffusion.noise.base import NoiseBase


@gin.configurable
class BrownianBridgeNoise(NoiseBase):
    """Brownian bridge noise schedule (from ETFlow)."""

    def __init__(self, eps: float = 1e-3, weight: float = 0.1):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def rate_noise(self, t):
        """Rate of change of noise."""
        numerator = 1 - 2 * t
        denominator = 2 * torch.sqrt(t * (1 - t) + self.eps)
        return numerator / denominator * self.weight

    def total_noise(self, t):
        """Total noise."""
        return torch.sqrt(t * (1 - t)) * self.weight
