"""Smooth LDDT-based loss that operates on Coordinates objects."""

import gin
import torch
from syncogen.diffusion.loss.base import LossBase
from syncogen.constants.constants import COORDS_STD


@gin.configurable
class SmoothLDDTLoss(LossBase):
    """Smooth LDDT-based loss."""

    def __init__(
        self,
        cutoff: float = 15.0,
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
        self.cutoff = cutoff
        self.sqrd = sqrd
        self.normalize_threshold = normalize_threshold

    def compute_loss(self, pred, target) -> torch.Tensor:
        """Base per-graph smooth LDDT loss (no time weighting or coef)."""
        C_true = target.atom_coords
        C_pred = pred.atom_coords
        coords_mask = target.atom_mask.bool()

        bs = C_true.shape[0]

        # Calculate pairwise distances
        true_dists = self._inter_distances(C_true, C_true)  # [bs, n_atoms, n_atoms]
        pred_dists = self._inter_distances(C_pred, C_pred)  # [bs, n_atoms, n_atoms]

        # Calculate distance differences
        dist_diff = torch.abs(true_dists - pred_dists)  # [bs, n_atoms, n_atoms]

        # SmoothLDDT thresholds
        lddt_thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=C_true.device)
        if self.normalize_threshold:
            lddt_thresholds = lddt_thresholds / COORDS_STD

        # Calculate epsilon values for each threshold
        eps = lddt_thresholds.reshape(1, 1, 1, -1) - dist_diff.unsqueeze(
            -1
        )  # [bs, n_atoms, n_atoms, n_thresholds]
        eps = torch.sigmoid(eps).mean(dim=-1)  # [bs, n_atoms, n_atoms]

        # Create inclusion radius mask (normalize cutoff if needed)
        cutoff = self.cutoff
        if self.normalize_threshold:
            cutoff = cutoff / COORDS_STD
        inclusion_mask = true_dists < cutoff

        # Remove self-interactions
        n_atoms = C_true.shape[1]
        mask = (
            inclusion_mask
            & ~torch.eye(n_atoms, dtype=torch.bool, device=C_true.device)[None, :, :]
        )

        # Apply coords mask if provided
        pairwise_mask = torch.stack(
            [torch.outer(coords_mask[i], coords_mask[i]) for i in range(bs)]
        )
        mask = mask & pairwise_mask

        # Calculate masked average
        mask_sum = mask.sum(dim=(-2, -1))  # [bs]
        lddt = (eps * mask).sum(dim=(-2, -1)) / (mask_sum + 1e-6)  # [bs]

        # Return 1.0 - lddt (loss increases as LDDT decreases)
        loss_per_graph = 1.0 - lddt
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
