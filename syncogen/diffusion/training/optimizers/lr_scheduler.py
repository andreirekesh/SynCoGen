from typing import Any, Dict

import gin
import torch


@gin.configurable
class LRScheduler:
    """Wrapper around torch/transformers LR schedulers for gin configuration."""

    def __init__(
        self,
        cls_name: str,
        module: str = "torch.optim.lr_scheduler",
        **kwargs: Dict[str, Any],
    ):
        self.cls_name = cls_name
        self.module = module
        self.kwargs = kwargs
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

    def initialize(self, optimizer: torch.optim.Optimizer):
        if self.module == "torch.optim.lr_scheduler":
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.cls_name)(
                optimizer, **self.kwargs
            )
        elif self.module == "transformers":
            import transformers

            self.lr_scheduler = getattr(transformers, self.cls_name)(
                optimizer, **self.kwargs
            )
        else:
            raise ValueError(f"Unknown module: {self.module}")

    def step(self):
        self.lr_scheduler.step()
