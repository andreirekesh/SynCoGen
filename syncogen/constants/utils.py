import torch

from syncogen.constants.constants import (
    FRAGMENT_MACCS,
    FRAGMENT_ATOMFEATS,
    FRAGMENT_ATOMADJ,
    FRAGMENT_BONDFEATS,
    N_BUILDING_BLOCKS,
    MAX_ATOMS_PER_BB,
    N_BOND_FEATURES,
)


def get_partial_maccs_keys(X_indices: torch.Tensor) -> torch.Tensor:
    """Get MACCS keys for a batch of molecules."""
    maccs_keys = FRAGMENT_MACCS.to(X_indices.device)[
        X_indices
    ]  # [batch_size, n_fragments, 166]
    return maccs_keys


def get_partial_atom_features(X_indices: torch.Tensor) -> torch.Tensor:
    """Get atom features for a batch of molecules."""
    atom_features = FRAGMENT_ATOMFEATS.to(X_indices.device)[
        X_indices
    ]  # [batch_size, n_fragments, MAX_ATOMS, 6]
    return atom_features


def get_partial_bond_features(X_indices, mode="adj"):
    """Constructs a batched adjacency matrix for the full atom graph from fragment adjacency matrices.

    Args:
        X_indices: Fragment indices tensor [BS, n_frags]
        mode: String indicating which features to use - must be "adj" or "feats"

    Returns:
        Adjacency matrix [BS, n_frags*max_atoms, n_frags*max_atoms, 5] where last dim is onehot:
        [single, double, triple, aromatic, is_masked]
    """
    if mode not in ["adj", "feats"]:
        raise ValueError("mode must be either 'adj' or 'feats'")

    batch_size = X_indices.shape[0]
    n_frags = X_indices.shape[1]
    total_atoms = n_frags * MAX_ATOMS_PER_BB

    # Initialize output adjacency matrix to all masked (onehot index 4)
    if mode == "adj":
        adj = torch.zeros(
            (batch_size, total_atoms, total_atoms), device=X_indices.device
        )
    elif mode == "feats":
        adj = torch.zeros(
            (batch_size, total_atoms, total_atoms, N_BOND_FEATURES),
            device=X_indices.device,
        )
        adj[..., -1] = 1  # Set is_masked=1 for all entries initially

    # Place fragment adjacency matrices along diagonal blocks
    for b in range(batch_size):
        for i in range(n_frags):
            frag_idx = X_indices[b, i]
            if frag_idx != N_BUILDING_BLOCKS:  # Not masked fragment
                start_idx = i * MAX_ATOMS_PER_BB
                end_idx = (i + 1) * MAX_ATOMS_PER_BB

                # Copy the fragment's adjacency matrix directly
                if mode == "adj":
                    adj[b, start_idx:end_idx, start_idx:end_idx] = FRAGMENT_ATOMADJ[
                        frag_idx
                    ]
                else:  # mode == "feats"
                    adj[b, start_idx:end_idx, start_idx:end_idx] = FRAGMENT_BONDFEATS[
                        frag_idx
                    ]

    return adj
