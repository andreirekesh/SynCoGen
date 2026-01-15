"""MSE losses for coordinate diffusion."""

import gin
import torch
from syncogen.diffusion.loss.base import LossBase


@gin.configurable
class MSELoss(LossBase):
    """Mean squared error loss for coordinates with optional time weighting.

    Uses LossBase.coef for weighting (no per-loss mse_coef).
    """

    def __init__(
        self,
        coef: float = 1.0,
        time_weighted: bool = False,
        square_time_weight: bool = False,
        t_threshold: float = None,
    ):
        super().__init__(
            mode="coords",
            coef=coef,
            time_weighted=time_weighted,
            square_time_weight=square_time_weight,
            t_threshold=t_threshold,
        )

    def compute_loss(self, pred, target) -> tuple[torch.Tensor, torch.Tensor]:
        """This is batchwise time-weighted MSE loss, but we can't do the full loss here because we need to time-weight by batch item."""
        C_pred = pred.atom_coords  # [B, A, 3]
        C_true = target.atom_coords  # [B, A, 3]
        mask = target.atom_mask.to(dtype=C_true.dtype)  # [B, A]
        mask3 = mask.unsqueeze(-1).expand_as(C_true)  # [B, A, 3]
        mse = (C_true - C_pred) ** 2  # [B, A, 3]
        numer = (mse * mask3).sum(dim=(1, 2))  # [B]
        denom = mask3.sum(dim=(1, 2))  # [B]
        # Return numer and denom with dimension [B] for time-weighting in the forward method
        return numer, denom

    def forward(self, pred, target, t: torch.Tensor = None) -> torch.Tensor:
        numer, denom = self.compute_loss(pred, target)  # [B], [B]
        dtype = numer.dtype

        if t is not None and self.t_threshold is not None:
            t_flat = t.reshape(-1).to(dtype=dtype)  # [B]
            sel = (t_flat <= self.t_threshold).to(dtype=dtype)  # [B]
            numer = numer * sel
            denom = denom * sel

        if t is not None and self.time_weighted:
            t_flat = t.reshape(-1).to(dtype=dtype)  # [B]
            tw = 1.0 / t_flat
            if self.square_time_weight:
                tw = tw**2
            numer = numer * tw

        denom_total = denom.sum().clamp_min(1)  # []
        loss = numer.sum() / denom_total  # []
        return self.apply_coef(loss)
