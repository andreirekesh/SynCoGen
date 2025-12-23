"""Base class for discrete sampling strategies."""

import torch
from typing import Tuple, Optional, Callable
from abc import ABC, abstractmethod

from syncogen.api.graph.graph import BBRxnGraph


class DiscreteStrategyBase(ABC):
    """Base class for discrete (graph) sampling strategies.

    Samplers accept BBRxnGraph objects and extract what they need internally.
    This keeps the call site clean and allows different strategies to access
    different properties (masks, indices, etc.) as needed.
    """

    def __init__(
        self,
        discrete_noise: Optional[Callable] = None,
        constrain_edge_sampling: bool = False,
    ):
        """
        Args:
            discrete_noise: Noise schedule function for discrete features.
            constrain_edge_sampling: Whether to constrain edge sampling step.
        """
        self.discrete_noise = discrete_noise
        self.constrain_edge_sampling = constrain_edge_sampling

    @staticmethod
    def node_mask_index(D_node: int) -> int:
        return D_node - 1

    @staticmethod
    def edge_mask_index(D_edge: int) -> int:
        return D_edge - 1

    @staticmethod
    def edge_no_edge_index(D_edge: int) -> int:
        return D_edge - 2

    @abstractmethod
    def step(
        self,
        graph: BBRxnGraph,
        p_x0: torch.Tensor,
        p_e0: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one denoising step for discrete features.
        Args:
            graph: BBRxnGraph containing current noisy discrete features
            p_x0: Predicted node probabilities (usually softmax/logits.exp) (B, N, D_node)
            p_e0: Predicted edge probabilities (B, N, N, D_edge)
            t: Current timestep (B, 1)
            dt: Time step size (optional, some strategies may require it)
        Returns:
            X_next: Updated node features (B, N, D_node)
            E_next: Updated edge features (B, N, N, D_edge)
        """
        raise NotImplementedError
