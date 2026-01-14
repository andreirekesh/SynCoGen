import copy
from abc import ABC, abstractmethod
import types

import numpy as np
import torch
import torch.nn as nn
import gin

from syncogen.constants.constants import (
    MAX_ATOMS_PER_BB,
    N_ATOM_FEATURES,
    N_BOND_FEATURES,
    N_BUILDING_BLOCKS,
    N_CENTERS,
    N_REACTIONS,
    N_PHARM,
)

from syncogen.models.layers import SinusoidalPositionalEmbedding
from syncogen.models.semla_utils import edges_from_nodes
from syncogen.constants.utils import (
    get_partial_atom_features,
    get_partial_bond_features,
    get_partial_maccs_keys,
)
from syncogen.api.ops.coordinates_ops import center
from syncogen.api.graph.graph import BBRxnGraph


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


# *************************************************************************************************
# *********************************** Helper Classes **********************************************
# *************************************************************************************************


class CoordNorm(torch.nn.Module):
    """Coordinate normalisation layer for coordinate sets with inductive bias towards molecules

    This layer allows 4 different types of coordinate normalisation (defined in the norm argument):
        1. 'none' - The coordinates are zero-centred and multiplied by learnable weights
        2. 'gvp' - Coords are zero-centred, scaled by learnable weights and each is scaled by sqrt(n_sets) / ||x_i||_2
        3. 'length' - Coords are zero-centred, multiplied by learnable weights and scaled by 1 / avg vector length

    Note that 'length' provides the same coordinate normalisation that is commonly used in current models but adapted
    to multiple coordinate sets, thereby allowing easier comparison to existing approaches.
    """

    def __init__(self, n_coord_sets, norm="length", eps=1e-6):
        super().__init__()

        norm = "none" if norm is None else norm
        if norm not in ["none", "gvp", "length"]:
            raise ValueError(f"Unknown normalisation type '{norm}'")

        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, coord_sets, node_mask, atom_only_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [batch_size, n_sets, n_nodes, 3]
        """

        # Zero the CoM in case it isn't already
        coord_sets = center(coord_sets, mask=atom_only_mask)
        coord_sets = coord_sets * node_mask.unsqueeze(-1)

        n_atoms = node_mask.sum(dim=-1, keepdim=True)
        lengths = torch.linalg.vector_norm(coord_sets, dim=-1)

        if self.norm == "length":
            scaled_lengths = lengths.sum(dim=2, keepdim=True) / n_atoms
            coord_div = scaled_lengths.unsqueeze(-1) + self.eps

        elif self.norm == "gvp":
            coord_div = (lengths.unsqueeze(-1) + self.eps) / np.sqrt(self.n_coord_sets)

        else:
            coord_div = torch.ones_like(coord_sets)

        coord_sets = (coord_sets * self.set_weights) / coord_div
        coord_sets = coord_sets * node_mask.unsqueeze(-1)
        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class EdgeMessages(torch.nn.Module):
    def __init__(
        self, d_model, d_message, d_out, n_coord_sets, d_ff=None, d_edge=None, eps=1e-6
    ):
        super().__init__()

        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff

        extra_feats = n_coord_sets + edge_feats
        in_feats = (d_message * 2) + extra_feats

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge) if d_edge is not None else None

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_out),
        )

    def forward(self, coords, node_feats, node_mask, atom_only_mask, edge_feats=None):
        """Compute edge messages

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            atom_only_mask (torch.Tensor): Mask for atoms only, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Edge messages tensor, shape [batch_size, n_nodes, n_nodes, d_out]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        node_feats = self.node_norm(node_feats)

        coords = self.coord_norm(coords, node_mask, atom_only_mask).flatten(0, 1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2))
        coord_feats = coord_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(
            1, -1
        )

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(node_feats)
        node_feats_start = node_feats.unsqueeze(2).expand(
            batch_size, n_nodes, n_nodes, -1
        )
        node_feats_end = node_feats.unsqueeze(1).expand(
            batch_size, n_nodes, n_nodes, -1
        )
        node_pairs = torch.cat((node_feats_start, node_feats_end), dim=-1)

        in_edge_feats = torch.cat((node_pairs, coord_feats), dim=3)
        if edge_feats is not None:
            edge_feats = self.edge_norm(edge_feats)
            in_edge_feats = torch.cat((in_edge_feats, edge_feats), dim=-1)

        return self.message_mlp(in_edge_feats)


class NodeAttention(torch.nn.Module):
    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()

        d_attn = d_model if d_attn is None else d_attn
        d_head = d_model // n_attn_heads

        if d_attn % n_attn_heads != 0:
            raise ValueError(
                "n_attn_heads must divide d_model (or d_attn if provided) exactly."
            )

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.feat_norm = torch.nn.LayerNorm(d_model)
        self.in_proj = torch.nn.Linear(d_model, d_attn)
        self.out_proj = torch.nn.Linear(d_attn, d_model)

    def forward(self, node_feats, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]

        Returns:
            torch.Tensor: Accumulated node features, shape [batch_size, n_nodes, d_model]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        node_feats = self.feat_norm(node_feats)
        proj_feats = self.in_proj(node_feats)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, n_nodes, d_head]
        # attentions shape [B * n_heads, n_nodes, n_nodes]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)

        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class CoordAttention(torch.nn.Module):
    def __init__(self, n_coord_sets, proj_sets=None, coord_norm="length", eps=1e-6):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, messages, adj_matrix, node_mask, atom_only_mask):
        """Compute an attention update for coordinate sets

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, proj_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            atom_only_mask (torch.Tensor): Mask for atoms only, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        coord_sets = self.coord_norm(coord_sets, node_mask, atom_only_mask)
        proj_coord_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # proj_coord_sets shape [B, 3, N, P]
        # norm_dists shape [B, 1, N, N, P]
        vec_dists = proj_coord_sets.unsqueeze(3) - proj_coord_sets.unsqueeze(2)
        lengths = torch.linalg.vector_norm(vec_dists, dim=1, keepdim=True)
        norm_dists = vec_dists / (lengths + self.eps)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Dim 1 is currently 1 on dists so we need to unsqueeze attentions
        updates = norm_dists * attentions.unsqueeze(1)
        updates = updates.sum(dim=3)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=2))
        updates = updates * weights.unsqueeze(1)

        # updates shape [B, 3, N, P] -> [B, S, N, 3]
        return self.attn_proj(updates).transpose(1, -1)


class LengthsMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()

        d_ff = d_model * 4 if d_ff is None else d_ff

        self.node_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(coord_sets, dim=-1).movedim(1, -1)
        in_feats = torch.cat((node_feats, lengths), dim=2)
        return self.node_ff(in_feats)


class EquivariantMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(proj_sets, proj_sets),
        )
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]
        inv_feats = self.node_proj(node_feats).unsqueeze(1)
        proj_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # Outer product with invariant features is equivariant, then sum over original coord sets
        attentions = inv_feats.unsqueeze(-1) * proj_sets.unsqueeze(-2)
        attentions = attentions.sum(-1)

        return self.attn_proj(attentions).transpose(1, -1)


class NodeFeedForward(torch.nn.Module):
    def __init__(
        self, d_model, n_coord_sets, d_ff=None, proj_sets=None, coord_norm="length"
    ):
        super().__init__()

        self.node_norm = torch.nn.LayerNorm(d_model)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)

        self.invariant_mlp = LengthsMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(
            d_model, n_coord_sets, proj_sets=proj_sets
        )

    def forward(self, coord_sets, node_feats, node_mask, atom_only_mask):
        """Pass data through the layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            atom_only_mask (torch.Tensor): Mask for atoms only, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """

        node_feats = self.node_norm(node_feats)
        coord_sets = self.coord_norm(coord_sets, node_mask, atom_only_mask)

        out_node_feats = self.invariant_mlp(coord_sets, node_feats)
        out_coord_sets = self.equivariant_mlp(coord_sets, node_feats)

        return out_coord_sets, out_node_feats


class BondRefine(torch.nn.Module):
    def __init__(self, d_model, d_message, d_edge, d_ff=None):
        super().__init__()

        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.coord_norm = CoordNorm(1, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_edge),
        )

    def forward(self, coords, node_feats, node_mask, atom_only_mask, edge_feats):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            coords (torch.Tensor): Coordinate tensor without coord sets, shape [batch_size, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_nodes], 1 for real, 0 otherwise
            atom_only_mask (torch.Tensor): Mask for atoms only, shape [batch_size, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Current edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Bond predictions tensor, shape [batch_size, n_nodes, n_nodes, n_bond_types]
        """

        assert len(coords.shape) == 3

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        # Calculate distances and dot products
        coords = self.coord_norm(
            coords.unsqueeze(1), node_mask.unsqueeze(1), atom_only_mask.unsqueeze(1)
        ).squeeze(1)
        coord_diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = (coord_diffs * coord_diffs).sum(dim=-1).unsqueeze(-1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2)).unsqueeze(-1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(self.node_norm(node_feats))
        node_feats_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_feats_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_i, node_feats_j), dim=-1)

        edge_feats = self.edge_norm(edge_feats)
        in_feats = torch.cat((node_pairs, dists, coord_dotprods, edge_feats), dim=3)
        return self.message_mlp(in_feats)


# *************************************************************************************************
# ********************************** Equivariant Layers *******************************************
# *************************************************************************************************


class EquiMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge_in=None,
        d_edge_out=None,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}"
            )

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_message_out = n_attn_heads + self.d_coord_message
        d_message_out = (
            d_message_out + d_edge_out if d_edge_out is not None else d_message_out
        )

        self.node_ff = NodeFeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
        )
        self.message_ff = EdgeMessages(
            d_model,
            d_message,
            d_message_out,
            n_coord_sets,
            d_ff=d_message_hidden,
            d_edge=d_edge_in,
            eps=eps,
        )
        self.coord_attn = CoordAttention(
            n_coord_sets, self.d_coord_message, coord_norm=coord_norm, eps=eps
        )
        self.node_attn = NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,
            "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(
        self, coords, node_feats, adj_matrix, node_mask, atom_only_mask, edge_feats=None
    ):
        """Pass data through the layer

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            atom_only_mask (torch.Tensor): Mask for atoms only, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge_in]

        Returns:
            Either a two-tuple of the new node coordinates and the new node features, or a three-tuple of the new
            node coords, new node features and new edge features.
        """

        if edge_feats is not None and self.d_edge_in is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge_in as None."
            )

        if edge_feats is None and self.d_edge_in is not None:
            raise ValueError(
                "The model was initialised with d_edge_in but no edge feats were provided to forward."
            )

        coord_updates, node_updates = self.node_ff(
            coords, node_feats, node_mask, atom_only_mask
        )
        coords = coords + coord_updates  # * atom_only_mask.unsqueeze(-1)
        node_feats = node_feats + node_updates
        ### ADD

        messages = self.message_ff(
            coords, node_feats, node_mask, atom_only_mask, edge_feats=edge_feats
        )
        node_messages = messages[:, :, :, : self.n_attn_heads]
        coord_messages = messages[
            :, :, :, self.n_attn_heads : (self.n_attn_heads + self.d_coord_message)
        ]
        node_feats = node_feats + self.node_attn(node_feats, node_messages, adj_matrix)
        coords = coords + self.coord_attn(
            coords, coord_messages, adj_matrix, node_mask, atom_only_mask
        )  # * atom_only_mask.unsqueeze(-1)

        if self.d_edge_out is not None:
            edge_out = messages[:, :, :, (self.n_attn_heads + self.d_coord_message) :]
            edge_out = edge_feats + edge_out if edge_feats is not None else edge_out
            return coords, node_feats, edge_out

        return coords, node_feats


# *************************************************************************************************
# ************************************* Dynamics Models *******************************************
# *************************************************************************************************


class EquiInvDynamics(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_layers,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge=None,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("n_layers is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}"
            )

        self._hparams = {
            "d_model": d_model,
            "d_message": d_message,
            "n_coord_sets": n_coord_sets,
            "n_layers": n_layers,
            "n_attn_heads": n_attn_heads,
            "d_message_hidden": d_message_hidden,
            "d_edge": d_edge,
            "bond_refine": bond_refine,
            "self_cond": self_cond,
            "coord_norm": coord_norm,
            "eps": eps,
        }

        self.d_model = d_model
        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.bond_refine = bond_refine and d_edge is not None
        self.self_cond = self_cond

        core_layer = EquiMessagePassingLayer(
            d_model,
            d_message,
            n_coord_sets,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            coord_norm=coord_norm,
            eps=eps,
        )
        layers = self._get_clones(core_layer, n_layers - extra_layers)

        if d_edge is not None:
            # Pass d_message_hidden as None so that these layers will have the same feats as their output
            in_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_in=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            out_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_out=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            layers = [in_layer] + layers + [out_layer]

        self.layers = torch.nn.ModuleList(layers)

        self.final_ff_block = NodeFeedForward(
            d_model, n_coord_sets, coord_norm=coord_norm
        )
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        # self.feat_norm = torch.nn.LayerNorm(d_model)

        in_coord_sets = 2 if self_cond else 1
        self.coord_proj = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.coord_head = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = torch.nn.LayerNorm(d_edge)

        if self.bond_refine:
            self.refine_layer = BondRefine(d_model, d_message, d_edge)

    @property
    def hparams(self):
        return self._hparams

    def forward(
        self,
        coords,
        inv_feats,
        adj_matrix,
        atom_mask=None,
        atom_and_pharm_mask=None,
        edge_feats=None,
        cond_coords=None,
    ):
        """Generate molecular coordinates and atom features

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for atom padding, shape [batch_size, n_atoms], 1 for real atoms
            atom_and_pharm_mask (torch.Tensor, Optional): Mask for atom/pharm padding, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]
            cond_coords (torch.Tensor, Optional): Conditional coords, shape [batch_size, n_nodes, 3]

        Returns:
            (coords, atom feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward."
            )

        if cond_coords is not None and not self.self_cond:
            raise ValueError(
                "cond_coords was provided but the model was initialised with self_cond as False."
            )

        if cond_coords is None and self.self_cond:
            raise ValueError(
                "The model was initialsed with self_cond but cond_coords was not provided."
            )

        # Project single coord set into a multiple learnable coord sets, while maintaining equivariance
        coords = (
            torch.stack((coords, cond_coords))
            if cond_coords is not None
            else coords.unsqueeze(0)
        )
        coords = self.coord_proj(coords.movedim(0, -1)).movedim(-1, 1)

        atom_mask = atom_mask.unsqueeze(1).expand(-1, self.n_coord_sets, -1)
        atom_and_pharm_mask = atom_and_pharm_mask.unsqueeze(1).expand(
            -1, self.n_coord_sets, -1
        )
        coords = coords * atom_and_pharm_mask.unsqueeze(-1)

        # Update coords and node feats using the model layers
        for layer in self.layers:
            out = layer(
                coords,
                inv_feats,
                adj_matrix,
                atom_and_pharm_mask,
                atom_mask,
                edge_feats=edge_feats,
            )
            if len(out) == 2:
                coords, inv_feats = out
                edge_feats = None
            elif len(out) == 3:
                coords, inv_feats, edge_feats = out

        # Apply a final feedforward block (also masked) and project coord sets to single coord set  ### CHANGED
        coords, inv_feats = self.final_ff_block(
            coords, inv_feats, atom_and_pharm_mask, atom_mask
        )

        out_coords = self.coord_norm(coords, atom_and_pharm_mask, atom_mask)
        out_coords = self.coord_head(out_coords.transpose(1, -1))
        out_coords = out_coords.transpose(1, -1).squeeze(1)

        if self.bond_refine:
            atom_mask = atom_mask[:, 0, :]
            atom_and_pharm_mask = atom_and_pharm_mask[:, 0, :]
            edge_feats = self.refine_layer(
                out_coords, inv_feats, atom_and_pharm_mask, atom_mask, edge_feats
            )

        # inv_feats = self.feat_norm(inv_feats)

        if self.d_edge is None:
            return out_coords, inv_feats

        edge_feats = self.bond_norm(edge_feats)
        return out_coords, inv_feats, edge_feats

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *********************************************************************************************************************
# ****************************************** Molecular Generation Models **********************************************
# *********************************************************************************************************************


@gin.configurable
class SemlaPharmGenerator(nn.Module):
    def __init__(
        self,
        d_model=384,
        d_message=128,
        n_coord_sets=64,
        n_layers=12,
        n_head=32,
        d_message_hidden=128,
        d_edge=128,
        coord_norm="length",
        self_conditioning=False,
        size_emb=64,
        pos_emb=64,
        length=5,
        pharmacophore_subset=40,
    ):
        super().__init__()
        self.self_cond = self_conditioning
        self.length = length
        self.pharmacophore_subset = pharmacophore_subset
        self.dynamics = EquiInvDynamics(
            d_model,
            d_message,
            n_coord_sets,
            n_layers,
            n_attn_heads=n_head,
            d_message_hidden=d_message_hidden,
            d_edge=d_edge,
            bond_refine=True,
            self_cond=self.self_cond,
            coord_norm=coord_norm,
        )

        self.self_cond = self_conditioning
        self.n_edge_types = N_REACTIONS * N_CENTERS * N_CENTERS + 2
        self.n_node_types = N_BUILDING_BLOCKS + 1
        self.n_pharm = N_PHARM + 1

        d_model_dim = d_model
        d_edge_dim = d_edge
        size_emb_dim = size_emb
        pos_emb_dim = pos_emb

        edge_in_feats = N_BOND_FEATURES * 2 if self.self_cond else N_BOND_FEATURES

        self.edge_in_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_in_feats, d_edge_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_edge_dim, d_edge_dim),
        )

        in_feats = N_ATOM_FEATURES * 2 if self.self_cond else N_ATOM_FEATURES
        in_feats = in_feats + size_emb_dim + pos_emb_dim + pos_emb_dim

        self.size_emb = torch.nn.Embedding(
            MAX_ATOMS_PER_BB * self.length + self.pharmacophore_subset + 1, size_emb_dim
        )
        self.pos_emb = SinusoidalPositionalEmbedding(
            pos_emb_dim, MAX_ATOMS_PER_BB * self.length + self.pharmacophore_subset
        )

        self.feat_proj = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_model_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model_dim, d_model_dim),
        )

        self.pharm_cond_proj = torch.nn.Sequential(
            torch.nn.Linear(self.n_pharm, d_model_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model_dim, N_ATOM_FEATURES),
        )

        if self.self_cond:
            self.X_cond_proj = torch.nn.Sequential(
                torch.nn.Linear(self.n_node_types, d_model_dim),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_model_dim, N_ATOM_FEATURES),
            )

            self.E_cond_proj = torch.nn.Sequential(
                torch.nn.Linear(self.n_edge_types, d_model_dim),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_model_dim, N_BOND_FEATURES),
            )

        self.X_atomtofrag_out = nn.Sequential(
            nn.Conv2d(
                in_channels=d_edge_dim,
                out_channels=d_edge_dim,
                kernel_size=MAX_ATOMS_PER_BB,
                stride=MAX_ATOMS_PER_BB,
                padding=0,
            ),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                in_channels=d_edge_dim,
                out_channels=d_edge_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.node_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_edge_dim * self.length + self.n_node_types, d_model_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model_dim, self.n_node_types),
        )

        self.E_atomtofrag_out = nn.Sequential(
            nn.Conv2d(
                in_channels=d_edge_dim,
                out_channels=d_edge_dim,
                kernel_size=MAX_ATOMS_PER_BB,
                stride=MAX_ATOMS_PER_BB,
                padding=0,
            ),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                in_channels=d_edge_dim,
                out_channels=d_edge_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.edge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_edge_dim + self.n_edge_types, d_edge_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_edge_dim, self.n_edge_types),
        )

    def forward(
        self,
        X,
        E,
        C,
        atom_mask,
        node_mask,
        pharm_types=None,  ### CHANGED
        pharm_pos=None,  ### CHANGED
        pharm_padding_mask=None,  ### CHANGED
    ):
        """Forward pass of the model.

        Args:
            X (torch.Tensor): Atom features tensor, shape [batch_size, n_fragments, n_atom_features]
            E (torch.Tensor): Edge features tensor, shape [batch_size, n_fragments, n_fragments, n_edge_features]
            C (torch.Tensor): Coordinates tensor, shape [batch_size, n_fragments * MAX_ATOMS_PER_BB, 3]
            atom_mask (torch.Tensor): Atom mask tensor, shape [batch_size, n_fragments * MAX_ATOMS_PER_BB]
            node_mask (torch.Tensor): Fragment mask tensor, shape [batch_size, n_fragments]
            pharm_types (torch.Tensor, Optional): Pharmacophore features, shape [batch_size, n_pharm, d_feat]  ### CHANGED
            pharm_pos (torch.Tensor, Optional): Pharmacophore coordinates, shape [batch_size, n_pharm, 3]  ### CHANGED
            pharm_padding_mask (torch.Tensor, Optional): Pharmacophore mask, shape [batch_size, n_pharm]  ### CHANGED

        Returns:
            tuple: Contains:
                - node_logits (torch.Tensor): Predicted node type logits, shape [batch_size, n_fragments, n_node_types]
                - edge_logits (torch.Tensor): Predicted edge type logits, shape [batch_size, n_fragments, n_fragments, n_edge_types]
                - pred_coords (torch.Tensor): Predicted coordinates, shape [batch_size, n_fragments * MAX_ATOMS_PER_BB, 3]
        """
        bs, n = X.shape[0], X.shape[1]
        pharm_types = pharm_types.to(X.dtype)

        if self.self_cond:
            X, X_cond, E, E_cond, C, C_cond = (
                X[..., : X.shape[-1] // 2],
                X[..., X.shape[-1] // 2 :],
                E[..., : E.shape[-1] // 2],
                E[..., E.shape[-1] // 2 :],
                C[..., : C.shape[-1] // 2],
                C[..., C.shape[-1] // 2 :],
            )
            X_cond_expanded = (
                X_cond.unsqueeze(2)
                .expand(-1, -1, MAX_ATOMS_PER_BB, -1)
                .reshape(bs, n * MAX_ATOMS_PER_BB, -1)
            )
            E_cond_expanded = (
                E_cond.unsqueeze(2)
                .unsqueeze(4)
                .expand(-1, -1, MAX_ATOMS_PER_BB, -1, MAX_ATOMS_PER_BB, -1)
                .reshape(bs, n * MAX_ATOMS_PER_BB, n * MAX_ATOMS_PER_BB, -1)
            )
            cond_atomics = self.X_cond_proj(X_cond_expanded)
            cond_bonds = self.E_cond_proj(E_cond_expanded)
            cond_coords = C_cond
        else:
            cond_atomics, cond_bonds, cond_coords = None, None, None

        # node_mask is the fragment-level mask (passed directly, no inference needed)
        frag_mask = node_mask.bool()
        n_fragments = frag_mask.sum(dim=-1)

        X_indices = X.argmax(dim=-1)

        cond_pharm = self.pharm_cond_proj(pharm_types)

        # concatenate pharmacophores as extra nodes if provided  ### CHANGED
        C = torch.cat([C, pharm_pos], dim=1)
        cond_coords = torch.cat([cond_coords, pharm_pos], dim=1)
        update_mask = torch.cat(
            [atom_mask, torch.zeros_like(pharm_padding_mask)], dim=1
        )
        atom_mask = torch.cat([atom_mask, pharm_padding_mask], dim=1)

        edge_feats = torch.zeros(
            (bs, n * MAX_ATOMS_PER_BB, n * MAX_ATOMS_PER_BB, 5), device=X.device
        )
        for i in range(bs):
            # Check if any nodes or edges are masked (last dim has 1 in last position)
            has_masked_nodes = X[i, : n_fragments[i], -1].any()
            has_masked_edges = E[i, : n_fragments[i], : n_fragments[i], -1].any()

            if not (has_masked_nodes or has_masked_edges):
                # No masks - try to compute exact bonds from graph structure
                try:
                    # Use frag_mask[i] as node_mask for this single sample
                    graph = BBRxnGraph.from_onehot(X[i], E[i], node_mask=frag_mask[i])
                    edge_feats[i] = graph.calculate_bonds(
                        reindex=False, as_onehot_adj_tensor=True
                    )
                except Exception:
                    # Fall back to partial bond features if calculation fails
                    edge_feats[i] = get_partial_bond_features(
                        X_indices[i : i + 1], mode="feats"
                    )[0]
            else:
                # Has masks - use partial bond features
                edge_feats[i] = get_partial_bond_features(
                    X_indices[i : i + 1], mode="feats"
                )[0]

        adj_matrix = edges_from_nodes(C, k=None, node_mask=atom_mask).to(torch.int8)

        atom_mask = atom_mask.to(torch.int8)
        update_mask = update_mask.to(torch.int8)

        inv_feats = get_partial_atom_features(X_indices)  # bs, n, MAX_ATOMS_PER_BB, 6
        inv_feats = inv_feats.reshape(bs, n * MAX_ATOMS_PER_BB, -1)

        pharm_types_padded = torch.zeros(
            (pharm_types.shape[0], pharm_types.shape[1], N_ATOM_FEATURES),
            device=inv_feats.device,
        )
        pharm_types_padded[:, :, : pharm_types.shape[-1]] = pharm_types
        inv_feats = torch.cat([inv_feats, pharm_types_padded], dim=1)

        # Embed the number of atoms in a mol into a small vector and concat this to inv feats for each atom
        n_atoms = atom_mask.sum(dim=-1, keepdim=True)

        size_emb = self.size_emb(n_atoms)
        size_emb = size_emb.expand(-1, inv_feats.size(1), -1)
        atom_positions = torch.arange(
            n * MAX_ATOMS_PER_BB + self.pharmacophore_subset, device=inv_feats.device
        ).int()
        frag_and_pharm_positions = torch.arange(n + 1, device=inv_feats.device).int()
        frag_positions = frag_and_pharm_positions[:-1]
        pharm_positions = frag_and_pharm_positions[-1:]

        # Generate embeddings
        atom_pos_emb = self.pos_emb(
            atom_positions
        )  # n*MAX_ATOMS_PER_BB + N_PHARM, pos_emb
        frag_pos_emb = self.pos_emb(frag_positions)  # n, pos_emb
        pharm_pos_emb = self.pos_emb(pharm_positions)

        # Expand fragment embeddings to atom dimensions
        frag_pos_emb = frag_pos_emb.unsqueeze(1).expand(
            -1, MAX_ATOMS_PER_BB, -1
        )  # n, MAX_ATOMS_PER_BB, pos_emb
        frag_pos_emb = frag_pos_emb.reshape(
            n * MAX_ATOMS_PER_BB, -1
        )  # n*MAX_ATOMS_PER_BB, pos_emb
        pharm_pos_emb = pharm_pos_emb.expand(
            self.pharmacophore_subset, pharm_pos_emb.shape[1]
        )  # pharmacophore_subset, pos_emb
        frag_pos_emb = torch.cat((frag_pos_emb, pharm_pos_emb), dim=0)

        # Expand atom embeddings to batch dimension
        atom_pos_emb = atom_pos_emb.unsqueeze(0).expand(
            bs, -1, -1
        )  # bs, n*MAX_ATOMS_PER_BB, pos_emb
        frag_pos_emb = frag_pos_emb.unsqueeze(0).expand(
            bs, -1, -1
        )  # bs, n*MAX_ATOMS_PER_BB, pos_emb

        # Concatenate all embeddings
        inv_feats = torch.cat((inv_feats, size_emb, atom_pos_emb, frag_pos_emb), dim=-1)

        if cond_atomics is not None:
            cond_pharm_and_atomics = torch.cat((cond_atomics, cond_pharm), dim=1)
            inv_feats = torch.cat((inv_feats, cond_pharm_and_atomics), dim=-1)

        atom_feats = self.feat_proj(inv_feats)

        if edge_feats is not None:
            edge_feats = edge_feats.to(torch.float16)
            edge_feats = (
                torch.cat((edge_feats, cond_bonds), dim=-1)
                if cond_bonds is not None
                else edge_feats
            )
            edge_feats_pharm = torch.zeros(
                (
                    edge_feats.shape[0],
                    edge_feats.shape[1] + self.pharmacophore_subset,
                    edge_feats.shape[2] + self.pharmacophore_subset,
                    edge_feats.shape[3],
                ),
                device=edge_feats.device,
            )
            edge_feats_pharm[:, :, :, -1] = 1
            edge_feats_pharm[:, : edge_feats.shape[1], : edge_feats.shape[2], :] = (
                edge_feats
            )
            edge_feats = self.edge_in_proj(edge_feats_pharm)

        out = self.dynamics(
            C,
            atom_feats,
            adj_matrix,
            atom_mask=update_mask,
            atom_and_pharm_mask=atom_mask,
            edge_feats=edge_feats,
            cond_coords=cond_coords,
        )
        pred_edges = None
        if len(out) == 2:
            pred_coords, pred_nodes = out
        elif len(out) == 3:
            pred_coords, pred_nodes, pred_edges = out

        # Make edges symmetric and permute once to (bs, channels, height, width) format
        pred_edges = pred_edges + pred_edges.transpose(1, 2)
        pred_edges = pred_edges.permute(0, 3, 1, 2)  # bs, d_edge, n_total, n_total

        # slice back to atoms only before classifier heads  ### CHANGED
        n_atoms_only = n * MAX_ATOMS_PER_BB
        pred_coords = pred_coords[:, :n_atoms_only, :]
        pred_nodes = pred_nodes[:, :n_atoms_only, :]
        pred_edges = pred_edges[:, :, :n_atoms_only, :n_atoms_only]
        atom_mask = atom_mask[:, :n_atoms_only]

        # Process nodes and edges in parallel using common permuted format
        node_feats = self.X_atomtofrag_out(pred_edges)  # bs, d_edge, n, n
        edge_feats = self.E_atomtofrag_out(pred_edges)  # bs, d_edge, n, n

        # Transform node features to final shape and classify
        node_feats = node_feats.permute(0, 2, 1, 3).contiguous()  # bs, n, d_edge, n
        node_feats = node_feats.flatten(2)  # bs, n, d_edge*n
        node_feats = torch.cat((node_feats, X), dim=-1)
        pred_nodes = self.node_classifier_head(node_feats)

        # Transform edge features to final shape and classify
        pred_edges = edge_feats.permute(0, 2, 3, 1).contiguous()  # bs, n, n, d_edge
        pred_edges = torch.cat((pred_edges, E), dim=-1)
        pred_edges = self.edge_classifier_head(pred_edges)

        # Coords
        pred_coords = center(pred_coords, mask=atom_mask)
        pred_edges = 1 / 2 * (pred_edges + torch.transpose(pred_edges, 1, 2))

        return pred_nodes, pred_edges, pred_coords
