"""Bond length preservation loss operating on Coordinates objects."""

import gin
import torch
from syncogen.diffusion.loss.base import LossBase


@gin.configurable
class BondLengthLoss(LossBase):
    """Bond length preservation loss computed from attached bonds.

    Expects pred and target as Coordinates. Assumes bonds (B, M, 3) and bonds_mask (B, M)
    are attached to either pred or target Coordinates. Uses LossBase.coef for weighting.
    """

    def __init__(
        self,
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
        self.sqrd = sqrd

    def compute_loss(self, pred, target) -> torch.Tensor:
        """Base per-graph bond-length loss (no time weighting or coef)."""
        bonds = None
        bonds_mask = None
        if getattr(target, "has_bonds", False):
            bonds = target.bonds
            bonds_mask = target.bonds_mask
        elif getattr(pred, "has_bonds", False):
            bonds = pred.bonds
            bonds_mask = pred.bonds_mask
        else:
            raise ValueError(
                "BondLengthLoss requires bonds attached to pred or target Coordinates"
            )

        # Flatten coordinates to atom dimension for flat indexing
        C_pred = pred.atom_coords
        C_true = target.atom_coords
        B = C_pred.shape[0]

        # Gather atom positions for each bond
        idx_i = bonds[..., 0].long()
        idx_j = bonds[..., 1].long()
        idx_i_exp = idx_i.unsqueeze(-1).expand(-1, -1, 3)
        idx_j_exp = idx_j.unsqueeze(-1).expand(-1, -1, 3)
        pred_i = torch.gather(C_pred, 1, idx_i_exp)
        pred_j = torch.gather(C_pred, 1, idx_j_exp)
        true_i = torch.gather(C_true, 1, idx_i_exp)
        true_j = torch.gather(C_true, 1, idx_j_exp)

        # Bond lengths
        pred_len = torch.linalg.norm(pred_i - pred_j, dim=-1)  # [B, M]
        true_len = torch.linalg.norm(true_i - true_j, dim=-1)  # [B, M]

        # Discrepancy per bond
        if self.sqrd:
            per_bond = (pred_len - true_len) ** 2
        else:
            per_bond = torch.abs(pred_len - true_len)

        # Apply bond mask
        if bonds_mask is not None:
            per_bond = per_bond * bonds_mask.to(per_bond.dtype)
            denom = bonds_mask.sum(dim=1).clamp_min(1)
        else:
            denom = torch.full(
                (B,), per_bond.shape[1], device=per_bond.device, dtype=per_bond.dtype
            )

        loss_per_graph = per_bond.sum(dim=1) / denom  # [B]
        return loss_per_graph

    def forward(self, pred, target, t: torch.Tensor = None) -> torch.Tensor:
        """Compute loss -> optional time weight -> threshold -> coef."""
        loss = self.compute_loss(pred, target)  # [B]
        if t is not None:
            loss = self.apply_time_weight(loss, t.reshape(-1, 1)).squeeze(-1)
        loss = self.apply_t_threshold(loss, t)
        return self.apply_coef(loss.mean())
