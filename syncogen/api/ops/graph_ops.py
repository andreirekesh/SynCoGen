"""Pure tensor operations for graph representation conversions.

All conversion functions:
- Accept (N, ...) for single graphs or (B, N, ...) for batched
- Return tensors of matching batch structure
- Are stateless and side-effect free
"""

import torch
from typing import Optional, List, Union, Tuple

# ============ Constants ============
# Channel indices for special tokens in onehots
NO_EDGE_CHANNEL = -2  # Penultimate channel indicates no edge
MASK_CHANNEL = -1  # Last channel indicates MASK token (for both reactions and BBs)


# ============ Edge Givens ============


def apply_edge_givens(rxn_onehot: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Apply edge invariants: diagonals are NO-EDGE, padding is zeros.

    This should be called after any operation that might violate these invariants
    (noising, sampling steps, initialization).

    Args:
        rxn_onehot: (N, N, D) or (B, N, N, D) reaction one-hot tensor
        node_mask: (N,) or (B, N) boolean/float mask where 1 = valid node

    Returns:
        rxn_onehot with invariants applied (modified in-place and returned)
    """
    is_batched = rxn_onehot.dim() == 4

    if is_batched:
        B, N, _, D = rxn_onehot.shape

        # 1. Diagonals -> NO-EDGE (for all samples)
        diag_idx = torch.arange(N, device=rxn_onehot.device)
        rxn_onehot[:, diag_idx, diag_idx, :] = 0
        rxn_onehot[:, diag_idx, diag_idx, NO_EDGE_CHANNEL] = 1

        # 2. Padding -> zeros (where either node is invalid)
        # edge_mask[b,i,j] = node_mask[b,i] AND node_mask[b,j]
        node_mask_float = node_mask.float() if node_mask.dtype == torch.bool else node_mask
        edge_valid = node_mask_float.unsqueeze(2) * node_mask_float.unsqueeze(1)  # (B, N, N)
        padding_mask = (edge_valid == 0).unsqueeze(-1)  # (B, N, N, 1)
        rxn_onehot = rxn_onehot.masked_fill(padding_mask, 0)
    else:
        N, _, D = rxn_onehot.shape

        # 1. Diagonals -> NO-EDGE
        diag_idx = torch.arange(N, device=rxn_onehot.device)
        rxn_onehot[diag_idx, diag_idx, :] = 0
        rxn_onehot[diag_idx, diag_idx, NO_EDGE_CHANNEL] = 1

        # 2. Padding -> zeros
        node_mask_float = node_mask.float() if node_mask.dtype == torch.bool else node_mask
        edge_valid = node_mask_float.unsqueeze(1) * node_mask_float.unsqueeze(0)  # (N, N)
        padding_mask = (edge_valid == 0).unsqueeze(-1)  # (N, N, 1)
        rxn_onehot = rxn_onehot.masked_fill(padding_mask, 0)

    return rxn_onehot


# ============ Initialization ============


def create_masked_graph(
    max_nodes: int,
    vocab_num_bbs: int,
    vocab_num_rxns: int,
    vocab_num_centers: int,
    bb_pad: int = 1,
    rxn_pad: int = 2,
    batch_size: Optional[int] = None,
    n_nodes: Optional[Union[int, List[int]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked graph one-hots and a node mask for padding.

    Convention:
    - Valid BBs: one-hot to MASK channel
    - Valid edges (off-diagonal): one-hot to MASK channel
    - Diagonals: one-hot to NO-EDGE channel (self-loops never exist)
    - Padding (BB or edge): all zeros

    Args:
        max_nodes: Maximum number of nodes (padding length)
        vocab_num_bbs: Building block vocabulary size
        vocab_num_rxns: Reaction vocabulary size
        vocab_num_centers: Number of reaction centers
        bb_pad: Extra dims for BB features (e.g., MASK token)
        rxn_pad: Extra dims for reaction features (e.g., NO-EDGE, MASK)
        batch_size: Optional batch size for batched initialization
        n_nodes: Actual number of nodes per sample (int for single, list for batched)
        device: Target device for tensors

    Returns:
        bb_onehot: (N, D_bb) or (B, N, D_bb) with MASK tokens for valid nodes
        rxn_onehot: (N, N, D_rxn) or (B, N, N, D_rxn) with MASK for valid edges, NO-EDGE for diagonals
        node_mask: (N,) or (B, N) indicating valid nodes
    """
    total_rxn_dim = vocab_num_rxns * vocab_num_centers * vocab_num_centers + rxn_pad
    bb_dim = vocab_num_bbs + bb_pad

    if batch_size is None:
        # Single graph
        if n_nodes is not None:
            assert isinstance(n_nodes, int), "n_nodes must be int for single graph"
        n_valid = n_nodes if n_nodes else max_nodes

        # Start with zeros (padding convention)
        bb = torch.zeros((max_nodes, bb_dim), dtype=torch.float32, device=device)
        rxn = torch.zeros((max_nodes, max_nodes, total_rxn_dim), dtype=torch.float32, device=device)
        node_mask = torch.zeros((max_nodes,), dtype=torch.float32, device=device)

        if n_valid > 0:
            node_mask[:n_valid] = 1.0
            bb[:n_valid, MASK_CHANNEL] = 1  # Valid nodes -> MASK
            rxn[:n_valid, :n_valid, MASK_CHANNEL] = 1  # Valid edges -> MASK (diagonals fixed below)

        # Apply edge givens: diagonals -> NO-EDGE, padding -> zeros
        rxn = apply_edge_givens(rxn, node_mask)
        return bb, rxn, node_mask

    # Batched
    bb = torch.zeros((batch_size, max_nodes, bb_dim), dtype=torch.float32, device=device)
    rxn = torch.zeros(
        (batch_size, max_nodes, max_nodes, total_rxn_dim), dtype=torch.float32, device=device
    )

    if n_nodes is None:
        node_mask = torch.ones((batch_size, max_nodes), dtype=torch.float32, device=device)
        bb[:, :, MASK_CHANNEL] = 1  # All nodes -> MASK
        rxn[:, :, :, MASK_CHANNEL] = 1  # All edges -> MASK (diagonals fixed below)
    else:
        lens = torch.as_tensor(n_nodes, dtype=torch.long)
        if lens.numel() != batch_size:
            raise ValueError("n_nodes must have length equal to batch_size")
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32, device=device)
        for b in range(batch_size):
            n = int(lens[b].item())
            if n > 0:
                node_mask[b, :n] = 1.0
                bb[b, :n, MASK_CHANNEL] = 1
                rxn[b, :n, :n, MASK_CHANNEL] = 1  # Valid edges -> MASK

    # Apply edge givens: diagonals -> NO-EDGE, padding -> zeros
    rxn = apply_edge_givens(rxn, node_mask)
    return bb, rxn, node_mask


# ============ BB Conversions ============


def bb_onehot_to_indices(bb_onehot: torch.Tensor) -> torch.Tensor:
    """Convert BB one-hot to indices.

    Args:
        bb_onehot: (N, D) or (B, N, D)

    Returns:
        (N,) or (B, N) tensor of indices
    """
    if bb_onehot.dim() == 2:
        return bb_onehot.argmax(dim=1)
    return bb_onehot.argmax(dim=2)


def bb_indices_to_onehot(bb_indices: torch.Tensor, vocab_size: int, pad: int = 1) -> torch.Tensor:
    """Convert BB indices to one-hot.

    Args:
        bb_indices: (N,) or (B, N) tensor of indices
        vocab_size: Number of building blocks in vocabulary
        pad: Number of padding dimensions (e.g., 1 for MASK token)

    Returns:
        (N, D) or (B, N, D) one-hot tensor where D = vocab_size + pad
    """
    total_dim = vocab_size + pad

    if bb_indices.dim() == 1:
        onehot = torch.zeros(
            (bb_indices.shape[0], total_dim), dtype=torch.float32, device=bb_indices.device
        )
        onehot.scatter_(1, bb_indices.unsqueeze(1).long(), 1.0)
        return onehot

    # Batched
    B, N = bb_indices.shape
    onehot = torch.zeros((B, N, total_dim), dtype=torch.float32, device=bb_indices.device)
    onehot.scatter_(2, bb_indices.unsqueeze(2).long(), 1.0)
    return onehot


# ============ Reaction Conversions ============


def rxn_onehot_to_indices(rxn_onehot: torch.Tensor) -> torch.Tensor:
    """Convert reaction one-hot to flat indices.

    Args:
        rxn_onehot: (N, N, D) or (B, N, N, D)

    Returns:
        (N, N) or (B, N, N) tensor of flat indices
    """
    if rxn_onehot.dim() == 3:
        return rxn_onehot.argmax(dim=2)
    return rxn_onehot.argmax(dim=3)


def rxn_indices_to_onehot(
    rxn_indices: torch.Tensor,
    vocab_num_rxns: int,
    vocab_num_centers: int,
    rxn_pad: int = 2,
) -> torch.Tensor:
    """Convert reaction flat indices to one-hot.

    Args:
        rxn_indices: (N, N) or (B, N, N) tensor of flat indices
        vocab_num_rxns: Number of reactions in vocabulary
        vocab_num_centers: Number of reaction centers
        rxn_pad: Number of padding dimensions (default 2 for NO-EDGE and PAD)

    Returns:
        (N, N, D) or (B, N, N, D) one-hot tensor
    """
    total_dim = vocab_num_rxns * vocab_num_centers * vocab_num_centers + rxn_pad

    if rxn_indices.dim() == 2:
        N = rxn_indices.shape[0]
        onehot = torch.zeros((N, N, total_dim), dtype=torch.float32, device=rxn_indices.device)
        onehot.scatter_(2, rxn_indices.unsqueeze(2).long(), 1.0)
        return onehot

    # Batched
    B, N, _ = rxn_indices.shape
    onehot = torch.zeros((B, N, N, total_dim), dtype=torch.float32, device=rxn_indices.device)
    onehot.scatter_(3, rxn_indices.unsqueeze(3).long(), 1.0)
    return onehot


def rxn_onehot_to_tuple(
    rxn_onehot: torch.Tensor, n_centers: int
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Convert reaction one-hot to tuple format.

    Args:
        rxn_onehot: (N, N, D) or (B, N, N, D)
        n_centers: Number of reaction centers

    Returns:
        Single: (E, 5) tensor with [rxn_id, node1, node2, center1, center2]
        Batched: List of (E_i, 5) tensors
    """
    if rxn_onehot.dim() == 3:
        return _rxn_onehot_to_tuple_single(rxn_onehot, n_centers)

    return [
        _rxn_onehot_to_tuple_single(rxn_onehot[b], n_centers) for b in range(rxn_onehot.shape[0])
    ]


def _rxn_onehot_to_tuple_single(rxn_onehot: torch.Tensor, n_centers: int) -> torch.Tensor:
    """Convert single reaction one-hot to tuple format."""
    # Find non-zero entries excluding NO-EDGE and PAD channels
    nonzero = torch.nonzero(rxn_onehot[..., :-2])
    # Keep only upper triangle (i < j) to avoid duplicates
    nonzero = nonzero[nonzero[:, 0] < nonzero[:, 1]]

    if len(nonzero) == 0:
        return torch.empty((0, 5), dtype=torch.long, device=rxn_onehot.device)

    flat_idx = nonzero[:, 2]
    centers_sq = n_centers**2
    reaction_idx = flat_idx // centers_sq
    center1_idx = (flat_idx % centers_sq) // n_centers
    center2_idx = flat_idx % n_centers

    result = torch.stack(
        [reaction_idx, nonzero[:, 0], nonzero[:, 1], center1_idx, center2_idx], dim=1
    )
    # Sort by node2 index for consistent reconstruction order
    sorted_indices = torch.argsort(result[:, 2])
    return result[sorted_indices]


def rxn_tuple_to_onehot(
    rxn_tuple: Union[torch.Tensor, List[torch.Tensor]],
    n_nodes: Union[int, List[int], torch.Tensor],
    vocab_num_rxns: int,
    vocab_num_centers: int,
    rxn_pad: int = 2,
) -> torch.Tensor:
    """Convert reaction tuples to one-hot matrices.

    Args:
        rxn_tuple: (E, 5) for single or List[(E_i, 5)] for batched
        n_nodes: Number of nodes (int for single, list/tensor for batched)
        vocab_num_rxns: Number of reactions in vocabulary
        vocab_num_centers: Number of reaction centers
        rxn_pad: Number of padding dimensions

    Returns:
        (N, N, D) or (B, N, N, D) one-hot tensor
    """
    # Single graph case
    if isinstance(n_nodes, int) or (torch.is_tensor(n_nodes) and n_nodes.dim() == 0):
        n = int(n_nodes) if isinstance(n_nodes, int) else int(n_nodes.item())
        return _rxn_tuple_to_onehot_single(rxn_tuple, n, vocab_num_rxns, vocab_num_centers, rxn_pad)

    # Batched case
    lens = torch.as_tensor(n_nodes, dtype=torch.long)
    B = lens.numel()

    if not isinstance(rxn_tuple, list):
        raise ValueError("For batched conversion, provide rxn_tuple as a list of per-graph tuples")
    assert len(rxn_tuple) == B, f"rxn_tuple list length {len(rxn_tuple)} != batch size {B}"

    outs = [
        _rxn_tuple_to_onehot_single(
            rxn_tuple[b], int(lens[b].item()), vocab_num_rxns, vocab_num_centers, rxn_pad
        )
        for b in range(B)
    ]
    return torch.stack(outs, dim=0)


def _rxn_tuple_to_onehot_single(
    rxn_tuple: torch.Tensor,
    n_nodes: int,
    vocab_num_rxns: int,
    vocab_num_centers: int,
    rxn_pad: int,
) -> torch.Tensor:
    """Convert single reaction tuple to one-hot matrix."""
    total_dim = vocab_num_rxns * vocab_num_centers * vocab_num_centers + rxn_pad
    centers_sq = vocab_num_centers**2

    onehot = torch.zeros(
        (n_nodes, n_nodes, total_dim), dtype=torch.float32, device=rxn_tuple.device
    )

    for rxn_data in rxn_tuple:
        reaction_idx, node1, node2, center1_idx, center2_idx = rxn_data.long()
        flat_idx = reaction_idx * centers_sq + center1_idx * vocab_num_centers + center2_idx
        onehot[node1, node2, flat_idx] = 1
        onehot[node2, node1, flat_idx] = 1  # Symmetric

    # Set NO-EDGE for positions without reactions
    has_edge = onehot[..., :-rxn_pad].any(dim=-1)
    onehot[..., NO_EDGE_CHANNEL][~has_edge] = 1

    return onehot


# ============ Compatibility Masks (unchanged) ============


def compute_compatibility_masks(
    bb_onehot: torch.Tensor,
    rxn_onehot: torch.Tensor,
    compatibility: torch.Tensor,
    no_edge_channel: int = -2,
):
    """Compute compatibility masks for nodes and edges given selected reactions and BBs.

    Inputs are one-hot tensors; outputs are boolean masks of the same shapes.

    Args:
        bb_onehot: (N, D_bb) or (B, N, D_bb)
        rxn_onehot: (N, N, D_rxn) or (B, N, N, D_rxn)
        compatibility: (N_bbs, N_rxns, N_centers) tensor with values in {0,1,2,3}
            0 = incompatible
            1 = compatible as reactant 1
            2 = compatible as reactant 2
            3 = compatible as both
        no_edge_channel: index of the NO-EDGE channel in rxn_onehot

    Returns:
        compatibility_mask_bb: bool tensor same shape as bb_onehot
        compatibility_mask_rxn: bool tensor same shape as rxn_onehot
    """
    single = bb_onehot.dim() == 2
    if single:
        bb = bb_onehot.unsqueeze(0)
        rxn = rxn_onehot.unsqueeze(0)
    else:
        bb = bb_onehot
        rxn = rxn_onehot

    B, N, D_bb = bb.shape
    _, _, _, D_rxn = rxn.shape

    # Default: everything is allowed
    comp_bb = torch.ones((B, N, D_bb), dtype=torch.bool, device=bb.device)
    comp_rxn = torch.ones((B, N, N, D_rxn), dtype=torch.bool, device=rxn.device)

    # Decode edge existence from NO-EDGE channel
    # Must also check that edge is valid (not all zeros from padding)
    rxn_idx = rxn.argmax(dim=-1)  # (B, N, N)
    no_edge_idx = D_rxn + no_edge_channel if no_edge_channel < 0 else no_edge_channel
    is_valid_edge = rxn.any(dim=-1)  # Padding edges are all zeros
    edge_exists = (rxn_idx < no_edge_idx) & is_valid_edge  # (B, N, N)

    # Compatibility codes per BB, reaction, and center
    comp_codes = compatibility.to(device=bb.device)  # (n_bbs, n_rxns, n_centers)
    n_bbs, n_rxns, n_centers = comp_codes.shape
    centers_sq = n_centers * n_centers
    comp_r1 = (comp_codes & 1).to(dtype=torch.bool)  # (n_bbs, n_rxns, n_centers)
    comp_r2 = (comp_codes & 2).to(dtype=torch.bool)  # (n_bbs, n_rxns, n_centers)

    # Reaction channel layout: [reactions * centers^2] + [NO-EDGE, PAD]
    core_dim = D_rxn - 2

    # ---- Node compatibility mask (per-node allowed BB channels) ----
    # For each edge (i,j) with i<j, we constrain node i in role 1 and node j in role 2.
    for b in range(B):
        for i in range(N):
            for j in range(i + 1, N):
                if not edge_exists[b, i, j]:
                    continue

                k = int(rxn_idx[b, i, j].item())
                if k >= core_dim:
                    continue  # NO-EDGE or PAD
                rxn_id = k // centers_sq
                within = k % centers_sq
                center1_idx = within // n_centers
                center2_idx = within % n_centers

                # Constraints for node i as reactant 1 at center1_idx
                mask_i_edge = torch.ones(D_bb, dtype=torch.bool, device=bb.device)
                mask_i_edge[:n_bbs] = comp_r1[:, rxn_id, center1_idx]
                cand_i = comp_bb[b, i] & mask_i_edge
                if cand_i.sum().item() > 1:
                    comp_bb[b, i] = cand_i

                # Constraints for node j as reactant 2 at center2_idx
                mask_j_edge = torch.ones(D_bb, dtype=torch.bool, device=bb.device)
                mask_j_edge[:n_bbs] = comp_r2[:, rxn_id, center2_idx]
                cand_j = comp_bb[b, j] & mask_j_edge
                if cand_j.sum().item() > 1:
                    comp_bb[b, j] = cand_j

    # ---- Edge compatibility mask (per-edge allowed reaction channels) ----
    # Determine current BB indices per node
    bb_idx = bb.argmax(dim=-1)  # (B, N)

    for b in range(B):
        for i in range(N):
            for j in range(i + 1, N):
                bb_i = int(bb_idx[b, i].item())
                bb_j = int(bb_idx[b, j].item())

                # If either endpoint is out-of-range (e.g., MASK token), leave this edge unconstrained
                if bb_i < 0 or bb_i >= n_bbs or bb_j < 0 or bb_j >= n_bbs:
                    continue

                # Build allowed mask across all reactions and center combinations at once
                allowed_all = torch.zeros(core_dim, dtype=torch.bool, device=bb.device)
                idx = 0
                for r in range(n_rxns):
                    for c1 in range(n_centers):
                        for c2 in range(n_centers):
                            if comp_r1[bb_i, r, c1] and comp_r2[bb_j, r, c2]:
                                allowed_all[idx] = True
                            idx += 1

                candidate = comp_rxn[b, i, j, :core_dim] & allowed_all
                # Only apply if more than one option remains, mirroring original guards
                if candidate.sum().item() > 1:
                    comp_rxn[b, i, j, :core_dim] = candidate
                    comp_rxn[b, j, i, :core_dim] = candidate  # symmetrize

    assert torch.all(comp_bb.sum(dim=-1) > 1)
    assert torch.all(comp_rxn.sum(dim=-1) > 1)

    if single:
        return comp_bb.squeeze(0), comp_rxn.squeeze(0)
    return comp_bb, comp_rxn
