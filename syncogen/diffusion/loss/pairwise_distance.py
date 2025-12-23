"""Pairwise distance loss for nearby atoms."""

import gin
import torch
from syncogen.diffusion.loss.base import LossBase


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
    ):
        super().__init__(
            mode="coords",
            coef=coef,
            time_weighted=time_weighted,
            square_time_weight=square_time_weight,
            t_threshold=t_threshold,
        )
        self.distance_threshold = distance_threshold
        self.sqrd = sqrd

    def compute_loss(self, pred, target) -> torch.Tensor:
        """Base per-graph pairwise distance loss (no time weighting or coef)."""
        C0 = target.atom_coords
        C1 = pred.atom_coords
        coords_mask = target.atom_mask.bool().tensor
        # Handle unbatched case [A, 3] -> [1, A, 3]
        if C0.dim() == 2:
            C0 = C0.unsqueeze(0)
            C1 = C1.unsqueeze(0)
            coords_mask = coords_mask.unsqueeze(0)

        bs = C0.shape[0]
        C0 = C0.reshape(bs, -1, 3)
        C1 = C1.reshape(bs, -1, 3)
        coords_mask = coords_mask.reshape(bs, -1)

        pairwise_mask = torch.stack(
            [torch.outer(coords_mask[i], coords_mask[i]) for i in range(bs)]
        )

        C0_dists = self._inter_distances(C0, C0)
        close_atoms_mask = (C0_dists < self.distance_threshold) & pairwise_mask

        C1_dists = self._inter_distances(C1, C1)
        loss = torch.abs(C1_dists - C0_dists)

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
        """Compute loss -> optional time weight -> coef."""
        loss = self.compute_loss(pred, target)  # [B]
        loss = self.apply_t_threshold(loss, t)
        if t is not None:
            loss = self.apply_time_weight(loss, t.reshape(-1, 1)).squeeze(-1)
        return self.apply_coef(loss.mean())

    def _inter_distances(self, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.sqrd:
            return torch.cdist(C1, C2, p=2) ** 2
        else:
            return torch.cdist(C1, C2, p=2)
