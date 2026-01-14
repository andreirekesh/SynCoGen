from typing import Any, Dict

import gin
import torch
from torch import nn


@gin.configurable
class Optimizer:
    """Wrapper around torch.optim optimizers for gin configuration."""

    def __init__(self, cls_name: str, **kwargs: Dict[str, Any]):
        self.cls_name = cls_name
        self.kwargs = kwargs
        self.optimizer: torch.optim.Optimizer = None

    def initialize(self, model: nn.Module):
        self.optimizer = getattr(torch.optim, self.cls_name)(
            model.parameters(), **self.kwargs
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
