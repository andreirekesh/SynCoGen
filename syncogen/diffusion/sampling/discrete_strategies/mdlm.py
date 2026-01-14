"""MDLM (Masked Diffusion Language Model) sampling strategy."""

import torch
from typing import Tuple
import gin
from syncogen.api.graph.graph import BBRxnGraph
from syncogen.api.ops.graph_ops import bb_indices_to_onehot, rxn_indices_to_onehot
from syncogen.diffusion.sampling.discrete_strategies.base import DiscreteStrategyBase
from syncogen.diffusion.sampling.discrete_strategies.utils import (
    sample_categorical,
    sample_edges,
)


@gin.configurable
class MDLM(DiscreteStrategyBase):
    """MDLM sampling strategy for discrete graph features.

    Uses the MDLM update rule: interpolate between staying masked
    and transitioning to the predicted distribution.
    """

    def __init__(
        self,
        discrete_noise=None,
        constrain_edge_sampling=True,
    ):
        super().__init__(discrete_noise, constrain_edge_sampling)

    def step(
        self,
        graph: BBRxnGraph,
        p_x0: torch.Tensor,
        p_e0: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one MDLM denoising step.

        Args:
            graph: BBRxnGraph with current noisy state
            p_x0: Predicted node probabilities (B, N, D_node)
            p_e0: Predicted edge probabilities (B, N, N, D_edge)
            t: Current timestep (B, 1)
            dt: Time step size
        Returns:
            X_next, E_next: Updated features
        """
        # Extract tensors from graph
        X = graph.bb_onehot
        E = graph.rxn_onehot

        if self.discrete_noise is None:
            raise ValueError("MDLM requires discrete_noise to be provided")
        node_mask_idx = self.node_mask_index(X.shape[-1])
        edge_mask_idx = self.edge_mask_index(E.shape[-1])

        # Get move chances from noise schedule
        discrete_sigma_t, _ = self.discrete_noise(t)
        discrete_sigma_s, _ = self.discrete_noise(t - dt)
        if discrete_sigma_t.ndim > 1:
            discrete_sigma_t = discrete_sigma_t.squeeze(-1)
        if discrete_sigma_s.ndim > 1:
            discrete_sigma_s = discrete_sigma_s.squeeze(-1)
        move_chance_t = 1 - torch.exp(-discrete_sigma_t)
        move_chance_s = 1 - torch.exp(-discrete_sigma_s)

        # Expand for broadcasting
        move_chance_t_X = move_chance_t[:, None, None]
        move_chance_t_E = move_chance_t[:, None, None, None]
        move_chance_s_X = move_chance_s[:, None, None]
        move_chance_s_E = move_chance_s[:, None, None, None]

        if self.constrain_edge_sampling:
            p_e0 = sample_edges(E, p_e0, graph.lengths)

        # MDLM update: q(x_s | x_t, x_0)
        # Probability of transitioning from masked to predicted
        q_xs = p_x0 * (move_chance_t_X - move_chance_s_X)
        q_xs[..., -1] = move_chance_s_X[..., 0]  # Probability of staying masked
        _X_indices, _ = sample_categorical(q_xs)

        q_es = p_e0 * (move_chance_t_E - move_chance_s_E)
        q_es[..., -1] = move_chance_s_E[..., 0]
        _E_indices, _ = sample_categorical(q_es)

        # Symmetrize edges and set diagonals to NO-EDGE index
        _E_upper = torch.triu(_E_indices, diagonal=1)
        _E_indices = _E_upper + _E_upper.transpose(1, 2)
        diag_indices = torch.arange(_E_indices.shape[1], device=_E_indices.device)
        _E_indices[:, diag_indices, diag_indices] = edge_mask_idx - 1  # NO-EDGE index

        # Convert to one-hot
        _X = bb_indices_to_onehot(
            _X_indices, vocab_size=BBRxnGraph.VOCAB_NUM_BBS, pad=1
        ).to(X.dtype)
        _E = rxn_indices_to_onehot(
            _E_indices,
            vocab_num_rxns=BBRxnGraph.VOCAB_NUM_RXNS,
            vocab_num_centers=BBRxnGraph.VOCAB_NUM_CENTERS,
            rxn_pad=2,
        ).to(E.dtype)

        # Copy flag: keep unmasked tokens, update masked ones
        # Unmasked = not the mask token
        copy_flag_X = (X.argmax(dim=-1) != node_mask_idx).unsqueeze(-1).float()
        copy_flag_E = (E.argmax(dim=-1) != edge_mask_idx).unsqueeze(-1).float()

        X_next = copy_flag_X * X + (1 - copy_flag_X) * _X
        E_next = copy_flag_E * E + (1 - copy_flag_E) * _E

        return X_next, E_next
