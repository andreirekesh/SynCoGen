import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch, to_dense_adj


def to_dense_graph(
    x,
    edge_index,
    edge_attr,
    batch,
    max_num_nodes,
):
    """Densify node and edge tensors and produce padding masks."""
    nodes_onehot, node_padding_mask = to_dense_batch(
        x=x, batch=batch, max_num_nodes=max_num_nodes
    )
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    edges_onehot = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )

    # Calculate edge mask from node mask
    edge_padding_mask = node_padding_mask.unsqueeze(-1) & node_padding_mask.unsqueeze(
        -2
    )
    # Make E symmetric by copying the upper triangle to the lower triangle (for first channel)
    edges_onehot[:, :, :, 0] = torch.triu(edges_onehot[:, :, :, 0]) + torch.triu(
        edges_onehot[:, :, :, 0], 1
    ).transpose(-1, -2)
    edges_onehot = encode_no_edge(edges_onehot)
    return nodes_onehot, edges_onehot, node_padding_mask, edge_padding_mask


def to_dense_coords(
    coordinates,
    coords_mask,
    batch,
    max_num_nodes,
):
    """Densify coordinates and coordinate masks if present."""
    coords = None
    coords_padding_mask = None
    if coordinates is not None and coords_mask is not None:
        coords, _ = to_dense_batch(
            x=coordinates, batch=batch, max_num_nodes=max_num_nodes
        )
        coords_padding_mask, _ = to_dense_batch(
            x=coords_mask, batch=batch, max_num_nodes=max_num_nodes
        )
    return coords, coords_padding_mask


def to_dense_bonds(
    bonds,
    bonds_len,
    max_num_bonds: int = None,
):
    """Densify variable-length per-graph bonds lists into a padded batch."""
    bonds_dense = None
    bonds_padding_mask = None
    if bonds is not None and bonds_len is not None:
        B = (
            bonds_len.shape[0]
            if isinstance(bonds_len, torch.Tensor)
            else len(bonds_len)
        )
        lens = torch.as_tensor(bonds_len, dtype=torch.long, device=bonds.device)
        assert lens.numel() == B, "bonds_len must have length equal to batch size"
        bonds_batch = torch.arange(B, device=bonds.device).repeat_interleave(lens)
        if max_num_bonds is None:
            max_num_bonds = int(lens.max().item()) if lens.numel() > 0 else 0
        bonds_dense, bonds_padding_mask = to_dense_batch(
            x=bonds, batch=bonds_batch, max_num_nodes=max_num_bonds
        )
    return bonds_dense, bonds_padding_mask


def to_dense_pharmacophores(
    pharm_types,
    pharm_pos,
    pharm_len,
    max_num_pharm: int = None,
):
    """Densify variable-length per-graph pharmacophores into padded batches."""
    pharm_types_dense = None
    pharm_pos_dense = None
    pharm_padding_mask = None
    if pharm_types is not None and pharm_len is not None:
        # Need pharm_len because pharm counts per-graph differ from node counts; cannot reuse node batch
        B = (
            pharm_len.shape[0]
            if isinstance(pharm_len, torch.Tensor)
            else len(pharm_len)
        )
        lens = torch.as_tensor(pharm_len, dtype=torch.long, device=pharm_types.device)
        assert lens.numel() == B, "pharm_len must have length equal to batch size"
        pharm_batch = torch.arange(B, device=pharm_types.device).repeat_interleave(lens)
        if max_num_pharm is None:
            max_num_pharm = int(lens.max().item()) if lens.numel() > 0 else 0
        pharm_types_dense, pharm_padding_mask = to_dense_batch(
            x=pharm_types, batch=pharm_batch, max_num_nodes=max_num_pharm
        )
        pharm_pos_dense, _ = to_dense_batch(
            x=pharm_pos, batch=pharm_batch, max_num_nodes=max_num_pharm
        )
    return pharm_types_dense, pharm_pos_dense, pharm_padding_mask


def encode_no_edge(edges_onehot: torch.Tensor) -> torch.Tensor:
    assert len(edges_onehot.shape) == 4
    if edges_onehot.shape[-1] == 0:
        return edges_onehot
    no_edge = torch.sum(edges_onehot, dim=3) == 0
    no_edge_idx = edges_onehot.shape[-1] - 2  # Second to last index
    no_edge_elt = edges_onehot[:, :, :, no_edge_idx]
    no_edge_elt[no_edge] = 1
    edges_onehot[:, :, :, no_edge_idx] = no_edge_elt
    diag = (
        torch.eye(edges_onehot.shape[1], dtype=torch.bool)
        .unsqueeze(0)
        .expand(edges_onehot.shape[0], -1, -1)
    )
    edges_onehot[diag] = 0
    return edges_onehot
