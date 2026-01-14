"""Negative log likelihood loss combining node and edge terms."""

import gin
import torch
from syncogen.diffusion.loss.base import LossBase


@gin.configurable
class NLLLoss(LossBase):
    """Combined masked NLL for nodes and edges.

    Expects log probabilities of the ground-truth class per node (B,N) and per edge (B,N,N).
    Supports optional time weighting; uses LossBase.coef to scale the sum.
    """

    def __init__(self, coef: float = 1.0):
        # Time-weighting disabled by design for NLL (uses sigma weighting instead)
        super().__init__(
            mode="graph", coef=coef, time_weighted=False, square_time_weight=False
        )

    def forward(
        self,
        log_p_theta_X: torch.Tensor,  # [B, N]
        log_p_theta_E: torch.Tensor,  # [B, N, N]
        node_padding_mask: torch.Tensor,  # [B, N] bool
        sigma_factor: torch.Tensor,  # [B] factor = dsigma / expm1(sigma)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive NLL components (nodes, edges), weighted by sigma_factor (mandatory).

        The caller (LossList) will aggregate these into the graph total.
        """
        # Apply sigma weighting to -log p
        loss_nodes = -log_p_theta_X * sigma_factor[:, None]
        loss_edges = -log_p_theta_E * sigma_factor[:, None, None]

        # Masked means
        masked_logp_nodes = loss_nodes * node_padding_mask
        denom_nodes = node_padding_mask.sum(dim=1).clamp_min(1)
        nll_nodes = masked_logp_nodes.sum(dim=1) / denom_nodes  # [B]

        # Edge NLL
        edge_padding_mask = node_padding_mask.unsqueeze(
            1
        ) & node_padding_mask.unsqueeze(
            2
        )  # [B,N,N]
        masked_logp_edges = loss_edges * edge_padding_mask
        denom_edges = edge_padding_mask.sum(dim=(1, 2)).clamp_min(1)  # [B]
        nll_edges = masked_logp_edges.sum(dim=(1, 2)) / denom_edges  # [B]

        # Apply coefficient to each component; LossList will sum
        node_loss = self.apply_coef(nll_nodes.mean())
        edge_loss = self.apply_coef(nll_edges.mean())
        # No additional time weighting here (disabled by design)
        return node_loss, edge_loss
