"""Base loss class for diffusion models."""

from typing import Literal, Optional, Sequence

import gin
import torch
import torch.nn as nn


LossMode = Literal["graph", "coords", "both"]


class LossBase(nn.Module):
    """Base class for all loss functions with built-in coefficient and optional time weighting."""

    def __init__(
        self,
        mode: LossMode = "both",
        coef: float = 1.0,
        time_weighted: bool = False,
        square_time_weight: bool = False,
        name: str = None,
        t_threshold: float = None,
    ):
        super().__init__()
        self.mode = mode
        self.coef = coef
        self.time_weighted = time_weighted
        self.square_time_weight = square_time_weight
        self._name = name
        # Optional time cutoff: apply loss only when t <= t_threshold
        self.t_threshold = t_threshold

    @property
    def name(self) -> str:
        """Return loss name. Defaults to class name if not explicitly set."""
        return self._name or self.__class__.__name__

    def apply_time_weight(
        self, loss: torch.Tensor, t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Optionally apply time weighting. Loss shapes are broadcast-compatible.
        NLLLoss should not call this (it has its own sigma weighting)."""
        if not self.time_weighted:
            return loss
        if t is None:
            raise ValueError("Time tensor t must be provided when time_weighted=True")
        if self.square_time_weight:
            time_weight = (1.0 / t) ** 2
        else:
            time_weight = 1.0 / t
        return loss * time_weight

    def apply_t_threshold(
        self, loss: torch.Tensor, t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Optionally mask loss to timesteps t <= t_threshold."""
        if self.t_threshold is None or t is None:
            return loss
        mask = (t <= self.t_threshold).to(loss.dtype)
        if mask.dim() == 0:
            mask = mask.unsqueeze(0)
        # Broadcast mask to match loss dimensions
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        return loss * mask

    def apply_coef(self, loss: torch.Tensor) -> torch.Tensor:
        return self.coef * loss

    def forward(self, *args, **kwargs):
        raise NotImplementedError


@gin.configurable
class LossList:
    """Aggregates multiple losses and returns dicts with individual values + totals."""

    def __init__(self, losses: Sequence[LossBase] = ()):
        self.losses = list(losses)

    def compute_graph(
        self, log_p_X, log_p_E, node_mask, sigma_factor
    ) -> dict[str, torch.Tensor]:
        result = {}
        total = 0.0
        for loss in self.losses:
            if loss.mode == "graph":
                value = loss.forward(log_p_X, log_p_E, node_mask, sigma_factor)
                # Allow graph losses to return either a single scalar tensor
                # or a tuple of (node_loss, edge_loss) for finer logging.
                if isinstance(value, tuple):
                    node_loss, edge_loss = value
                    result[f"{loss.name}_nodes"] = node_loss
                    result[f"{loss.name}_edges"] = edge_loss
                    combined = node_loss + edge_loss
                    result[f"{loss.name}"] = combined
                    total += combined
                else:
                    result[f"{loss.name}"] = value
                    total += value
        result["graph_total"] = total
        return result

    def compute_coords(self, coords_pred, coords_gt, t) -> dict[str, torch.Tensor]:
        result = {}
        total = 0.0
        for loss in self.losses:
            if loss.mode == "coords":
                value = loss.forward(coords_pred, coords_gt, t)
                result[f"{loss.name}"] = value
                total += value
        result["coords_total"] = total
        return result
