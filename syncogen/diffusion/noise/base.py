"""Base noise schedule class."""

import abc
import torch
import torch.nn as nn


class NoiseBase(abc.ABC, nn.Module):
    """Base noise schedule class."""

    def forward(self, t):
        """Get total and rate of noise at timestep t."""
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """Rate of change of noise (g(t))."""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """Total noise (integral of g(t) from 0 to t + g(0))."""
        pass
