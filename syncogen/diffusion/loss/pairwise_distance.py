"""Pairwise distance loss for nearby atoms."""

from numpy import True_
import gin
import torch
from syncogen.diffusion.loss.base import LossBase
from syncogen.constants.constants import COORDS_STD


@gin.configurable
class PairwiseDistanceLoss(LossBase):
    """Pairwise distance loss for nearby atoms."""

    def __init__(
        self,
        distance_threshold: float = 5.0,
        sqrd: bool = False,
        coef: float = 1.0,
        time_weighted: bool = False,
        square_time_weight: bool = False,
        t_threshold: float = None,
        normalize_threshold: bool = True,
    ):
        super().__init__(
            mode="coords",
            coef=coef,
            time_weighted=time_weighted,
            square_time_weight=square_time_weight,
            t_threshold=t_threshold,
        )
        self.distance_threshold = distance_threshold
        if normalize_threshold:
            self.distance_threshold = self.distance_threshold / COORDS_STD
        self.sqrd = sqrd

    def compute_loss(self, pred, target) -> torch.Tensor:
        """Base per-graph pairwise distance loss (no time weighting or coef).

        Uses target.atom_mask (ground truth mask) to only consider distances between real atoms.
        """
        C_true = target.atom_coords
        C_pred = pred.atom_coords
        coords_mask = target.atom_mask.bool()  # Uses ground truth mask

        bs = C_true.shape[0]
        pairwise_mask_true = torch.stack(
            [torch.outer(coords_mask[i], coords_mask[i]) for i in range(bs)]
        )

        C_dists = self._inter_distances(C_true, C_true)
        close_atoms_mask = (C_dists < self.distance_threshold) & pairwise_mask_true

        C_dists_pred = self._inter_distances(C_pred, C_pred)
        loss = torch.abs(C_dists_pred - C_dists)

        masked_loss = loss * close_atoms_mask
        total = masked_loss.sum(dim=(-2, -1))
        count = close_atoms_mask.sum(dim=(-2, -1))
        loss_per_graph = total / (count + 1e-6)
        return loss_per_graph

    def forward(
        self,
        pred,
        target,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute loss -> optional time weight -> threshold -> coef."""
        loss = self.compute_loss(pred, target)  # [B]
        if t is not None:
            loss = self.apply_time_weight(loss, t.reshape(-1, 1)).squeeze(-1)
        loss = self.apply_t_threshold(loss, t)
        return self.apply_coef(loss.mean())

    def _inter_distances(self, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.sqrd:
            return torch.cdist(C1, C2, p=2) ** 2
        else:
            return torch.cdist(C1, C2, p=2)
