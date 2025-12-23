"""P2 (Path Planning) discrete sampling strategy.

Path planning uses confidence-based or random scoring to decide which tokens
to unmask at each step, allowing more control over the denoising trajectory.
"""

import torch
import gin
from typing import Tuple, Literal

from syncogen.api.graph.graph import BBRxnGraph
from syncogen.api.ops.graph_ops import bb_indices_to_onehot, rxn_indices_to_onehot
from syncogen.diffusion.sampling.discrete_strategies.base import DiscreteStrategyBase
from syncogen.diffusion.sampling.discrete_strategies.utils import sample_categorical


@gin.configurable
class PathPlanning(DiscreteStrategyBase):
    """Path Planning sampling strategy for discrete graph features.

    Instead of using the MDLM probabilistic update, path planning:
    1. Samples predictions from logits/probabilities
    2. Scores each token by confidence (or randomly)
    3. Uses top-k masking to decide which tokens to remask
    4. Reveals tokens that were masked but shouldn't be anymore

    This provides more control over the generation trajectory.
    """

    def __init__(
        self,
        discrete_noise=None,
        constrain_edge_sampling=True,
        score_type: Literal["confidence", "random"] = "confidence",
        temperature: float = 1.0,
        eta: float = 1.0,
    ):
        super().__init__(discrete_noise, constrain_edge_sampling)
        self.score_type = score_type
        self.temperature = temperature
        self.eta = eta

    def _topk_lowest_masking(self, scores: torch.Tensor, cutoff_len: torch.Tensor) -> torch.Tensor:
        """Select tokens with the lowest scores up to cutoff_len per batch.

        Args:
            scores: (B, N) or (B, N*N) scores where lower = more likely to mask
            cutoff_len: (B, 1) number of tokens to mask per batch

        Returns:
            Boolean mask where True = should be masked
        """
        sorted_scores, _ = scores.sort(dim=-1)
        # Clamp cutoff_len to valid range
        cutoff_len = cutoff_len.clamp(0, scores.shape[-1] - 1)
        threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
        return scores < threshold

    def step(
        self,
        graph: BBRxnGraph,
        p_x0: torch.Tensor,
        p_e0: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one path planning denoising step.

        Args:
            graph: BBRxnGraph with current noisy state
            p_x0: Predicted node probabilities (B, N, D_node)
            p_e0: Predicted edge probabilities (B, N, N, D_edge)
            t: Current timestep (B, 1)
            dt: Time step size (unused by PathPlanning, but accepted for API consistency)

        Returns:
            X_next, E_next: Updated features
        """
        # Extract from graph
        X = graph.bb_onehot
        E = graph.rxn_onehot
        node_mask = graph.node_mask
        edge_mask = graph.edge_mask if hasattr(graph, "edge_mask") else None

        if self.discrete_noise is None:
            raise ValueError("PathPlanning requires discrete_noise to be provided")

        # Get move chance from noise schedule
        discrete_sigma_t, _ = self.discrete_noise(t)
        move_chance_t = 1 - torch.exp(-discrete_sigma_t)

        # ===== NODES =====
        X_next = self._step_nodes(X, p_x0, move_chance_t, node_mask)

        # ===== EDGES =====
        E_next = self._step_edges(E, p_e0, move_chance_t, edge_mask)

        return X_next, E_next

    def _step_nodes(
        self,
        X: torch.Tensor,
        p_x0: torch.Tensor,
        move_chance_t: torch.Tensor,
        node_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Path planning step for nodes."""
        B, N, D = X.shape
        device = X.device
        node_mask_idx = self.node_mask_index(D)

        # Get current mask status
        X_indices = X.argmax(dim=-1)
        is_masked_X = X_indices == node_mask_idx
        is_unmasked_X = ~is_masked_X

        # Sample predictions from probability tensor
        probs_X = p_x0
        # Respect temperature for sampling if not 1.0
        if self.temperature != 1.0:
            probs_X = probs_X ** (1.0 / self.temperature)
            probs_X = probs_X / probs_X.sum(dim=-1, keepdim=True)
        x0_indices, log_probs_X = sample_categorical(
            probs_X, temperature=1.0
        )  # temperature already applied

        # Compute scores for masking decisions
        if self.score_type == "confidence":
            # Use max log probability as confidence score
            score_X = log_probs_X.max(dim=-1)[0]
        else:  # random
            score_X = torch.rand(B, N, device=device).log()

        # Apply padding mask to scores (pad -> inf so they're never selected for masking)
        if node_mask is not None:
            score_X = score_X.masked_fill(~node_mask.bool(), float("inf"))

        # Eta scaling: unmasked tokens get scaled scores (higher eta = harder to remask)
        score_X = torch.where(is_unmasked_X, score_X * self.eta, score_X)

        # Calculate how many tokens to mask based on schedule
        if node_mask is not None:
            num_valid = node_mask.sum(dim=1, keepdim=True).float()
        else:
            num_valid = torch.full((B, 1), N, device=device, dtype=torch.float)
        num_to_mask_X = (num_valid * move_chance_t).long()

        # Select tokens with lowest scores to mask
        should_mask_X = self._topk_lowest_masking(score_X, num_to_mask_X)

        # Build new indices:
        # - If should_mask: set to mask index
        # - If was masked but shouldn't be: reveal (use x0 prediction)
        # - Otherwise: keep current
        X_next_indices = X_indices.clone()
        X_next_indices[should_mask_X] = node_mask_idx

        # Reveal: was masked AND not selected to stay masked
        reveal_X = is_masked_X & ~should_mask_X
        X_next_indices[reveal_X] = x0_indices[reveal_X]

        # Convert back to one-hot
        X_next = bb_indices_to_onehot(
            X_next_indices, vocab_size=BBRxnGraph.VOCAB_NUM_BBS, pad=1
        ).to(X.dtype)

        return X_next

    def _step_edges(
        self,
        E: torch.Tensor,
        p_e0: torch.Tensor,
        move_chance_t: torch.Tensor,
        edge_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Path planning step for edges."""
        B, N, _, D = E.shape
        device = E.device
        edge_mask_idx = self.edge_mask_index(D)

        # Get current mask status
        E_indices = E.argmax(dim=-1)
        is_masked_E = E_indices == edge_mask_idx
        is_unmasked_E = ~is_masked_E

        # Sample predictions from probability tensor
        probs_E = p_e0
        if self.temperature != 1.0:
            probs_E = probs_E ** (1.0 / self.temperature)
            probs_E = probs_E / probs_E.sum(dim=-1, keepdim=True)
        e0_indices, log_probs_E = sample_categorical(
            probs_E, temperature=1.0
        )  # temperature already applied

        # Symmetrize sampled indices (diagonals handled by apply_edge_givens in sampling loop)
        e0_indices_upper = torch.triu(e0_indices, diagonal=1)
        e0_indices = e0_indices_upper + e0_indices_upper.transpose(1, 2)

        # Compute scores
        if self.score_type == "confidence":
            score_E = log_probs_E.max(dim=-1)[0]
        else:
            score_E = torch.rand(B, N, N, device=device).log()

        # Apply padding mask
        if edge_mask is not None:
            score_E = score_E.masked_fill(~edge_mask.bool(), float("inf"))

        # Eta scaling for unmasked edges
        score_E = torch.where(is_unmasked_E, score_E * self.eta, score_E)

        # Calculate how many edges to mask (count upper triangle only, then double for symmetry)
        if edge_mask is not None:
            num_valid = edge_mask.triu(diagonal=1).sum(dim=(1, 2), keepdim=True).squeeze(2).float()
        else:
            num_valid = torch.full((B, 1), N * (N - 1) // 2, device=device, dtype=torch.float)
        num_to_mask_E = (2 * num_valid * move_chance_t).long()

        # Flatten for top-k selection
        score_E_flat = score_E.view(B, -1)
        should_mask_E = self._topk_lowest_masking(score_E_flat, num_to_mask_E).view(B, N, N)

        # Build new indices
        E_next_indices = E_indices.clone()
        E_next_indices[should_mask_E] = edge_mask_idx

        # Reveal masked edges that shouldn't stay masked
        reveal_E = is_masked_E & ~should_mask_E
        E_next_indices[reveal_E] = e0_indices[reveal_E]

        # Convert back to one-hot
        E_next = rxn_indices_to_onehot(
            E_next_indices,
            vocab_num_rxns=BBRxnGraph.VOCAB_NUM_RXNS,
            vocab_num_centers=BBRxnGraph.VOCAB_NUM_CENTERS,
            rxn_pad=2,
        ).to(E.dtype)

        return E_next
