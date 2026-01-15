"""Building Block and Reaction Graph representation.

This module provides BBRxnGraph, a graph representation that stores both
building blocks (nodes) and reactions (edges) in one-hot format.

The canonical representation is one-hot tensors. Derived representations
(indices, tuples) are computed lazily and cached.
"""

import gin
import torch
from typing import Optional, List, Union, Tuple

from syncogen.api.graph.building_block import BuildingBlock
from syncogen.api.graph.reaction import Reaction
from syncogen.api.ops.graph_ops import (
    create_masked_graph,
    apply_edge_givens,
    bb_onehot_to_indices,
    bb_indices_to_onehot,
    rxn_onehot_to_indices,
    rxn_onehot_to_tuple,
    rxn_indices_to_onehot,
    rxn_tuple_to_onehot,
    compute_compatibility_masks,
)
from syncogen.constants.constants import (
    MAX_ATOMS_PER_BB,
    N_CENTERS,
    N_REACTIONS,
    N_BUILDING_BLOCKS,
    COMPATIBILITY,
)


class BBRxnGraph:
    """Building block and reaction graph.

    Stores graphs in one-hot format as the canonical representation.
    Derived representations (indices, tuples) are computed lazily and cached.

    Use factory methods for construction:
        - BBRxnGraph.from_onehot(bb_onehot, rxn_onehot, node_mask) - node_mask required
        - BBRxnGraph.from_indices(bb_indices, rxn_indices) - assumes all valid if no mask
        - BBRxnGraph.from_tuple(bb_indices, rxn_tuple) - assumes all valid if no mask
        - BBRxnGraph.masked(max_nodes, ...) - creates node_mask from n_nodes

    Properties automatically update when bb_onehot or rxn_onehot are modified.
    """

    # Class-level vocabulary constants
    VOCAB_NUM_BBS = N_BUILDING_BLOCKS
    VOCAB_NUM_RXNS = N_REACTIONS
    VOCAB_NUM_CENTERS = N_CENTERS

    def __init__(
        self,
        bb_onehot: torch.Tensor,
        rxn_onehot: torch.Tensor,
        node_mask: torch.Tensor,
        bb_pad: int = 1,
        rxn_pad: int = 2,
    ):
        """Initialize from one-hot tensors (canonical form).

        Prefer using factory methods instead of calling __init__ directly.

        Args:
            bb_onehot: (N, D_bb) or (B, N, D_bb) building block one-hot
            rxn_onehot: (N, N, D_rxn) or (B, N, N, D_rxn) reaction one-hot
            node_mask: Optional (N,) or (B, N) mask for valid nodes
            bb_pad: Number of padding dims in BB one-hot (default 1 for MASK)
            rxn_pad: Number of padding dims in rxn one-hot (default 2 for NO-EDGE, PAD)
        """
        # Primary tensor defines device/dtype
        self._bb_onehot = bb_onehot
        device, dtype = bb_onehot.device, bb_onehot.dtype

        # Move secondary tensors to match primary
        self._rxn_onehot = rxn_onehot.to(device=device, dtype=dtype)
        self._node_mask = node_mask.to(device=device, dtype=bool)

        self.bb_pad = bb_pad
        self.rxn_pad = rxn_pad

        # Infer batched mode from tensor shapes
        self._is_batched = bb_onehot.dim() == 3

        # Invalidatable caches for derived representations
        self._bb_indices = None
        self._rxn_indices = None
        self._rxn_tuple = None
        self._building_blocks = None
        self._reactions = None

    # ============ Factory Methods ============

    @classmethod
    def from_onehot(
        cls,
        bb_onehot: torch.Tensor,
        rxn_onehot: torch.Tensor,
        node_mask: torch.Tensor,
        bb_pad: int = 1,
        rxn_pad: int = 2,
    ) -> "BBRxnGraph":
        """Construct from one-hot tensors.

        Args:
            bb_onehot: (N, D_bb) or (B, N, D_bb) building block one-hot
            rxn_onehot: (N, N, D_rxn) or (B, N, N, D_rxn) reaction one-hot
            node_mask: (N,) or (B, N) mask for valid nodes (REQUIRED)
            bb_pad: Number of padding dims in BB one-hot
            rxn_pad: Number of padding dims in rxn one-hot
        """
        return cls(bb_onehot, rxn_onehot, node_mask, bb_pad, rxn_pad)

    @classmethod
    def from_indices(
        cls,
        bb_indices: torch.Tensor,
        rxn_indices: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        bb_pad: int = 1,
        rxn_pad: int = 2,
    ) -> "BBRxnGraph":
        """Construct from index tensors.

        Args:
            bb_indices: (N,) or (B, N) building block indices
            rxn_indices: (N, N) or (B, N, N) flat reaction indices
            node_mask: Optional (N,) or (B, N) mask. If None, assumes all nodes valid.
            bb_pad: Number of padding dims in BB one-hot
            rxn_pad: Number of padding dims in rxn one-hot
        """
        bb_onehot = bb_indices_to_onehot(bb_indices, cls.VOCAB_NUM_BBS, bb_pad)
        rxn_onehot = rxn_indices_to_onehot(
            rxn_indices, cls.VOCAB_NUM_RXNS, cls.VOCAB_NUM_CENTERS, rxn_pad
        )
        if node_mask is None:
            # Default: all nodes valid (for non-padded graphs)
            node_mask = torch.ones(
                bb_indices.shape, dtype=torch.bool, device=bb_indices.device
            )
        return cls(bb_onehot, rxn_onehot, node_mask, bb_pad, rxn_pad)

    @classmethod
    def from_tuple(
        cls,
        bb_indices: torch.Tensor,
        rxn_tuple: Union[torch.Tensor, List[torch.Tensor]],
        node_mask: Optional[torch.Tensor] = None,
        bb_pad: int = 1,
        rxn_pad: int = 2,
    ) -> "BBRxnGraph":
        """Construct from BB indices and reaction tuples.

        Args:
            bb_indices: (N,) or (B, N) building block indices
            rxn_tuple: (E, 5) or List[(E_i, 5)] reaction tuples
                       Each tuple: [rxn_id, node1, node2, center1, center2]
            node_mask: Optional (N,) or (B, N) mask. If None, assumes all nodes valid.
            bb_pad: Number of padding dims in BB one-hot
            rxn_pad: Number of padding dims in rxn one-hot
        """
        bb_onehot = bb_indices_to_onehot(bb_indices, cls.VOCAB_NUM_BBS, bb_pad)

        # Determine n_nodes for rxn_tuple conversion
        if bb_indices.dim() == 1:
            n_nodes = bb_indices.shape[0]
        else:
            n_nodes = [bb_indices.shape[1]] * bb_indices.shape[0]

        rxn_onehot = rxn_tuple_to_onehot(
            rxn_tuple, n_nodes, cls.VOCAB_NUM_RXNS, cls.VOCAB_NUM_CENTERS, rxn_pad
        )
        if node_mask is None:
            # Default: all nodes valid (for non-padded graphs)
            node_mask = torch.ones(
                bb_indices.shape, dtype=torch.bool, device=bb_indices.device
            )
        return cls(bb_onehot, rxn_onehot, node_mask, bb_pad, rxn_pad)

    @classmethod
    def masked(
        cls,
        max_nodes: int,
        batch_size: Optional[int] = None,
        n_nodes: Optional[Union[int, List[int]]] = None,
        bb_pad: int = 1,
        rxn_pad: int = 2,
        device: Optional[torch.device] = None,
    ) -> "BBRxnGraph":
        """Construct a fully masked graph.

        Args:
            max_nodes: Maximum number of nodes (padding length)
            batch_size: Optional batch size for batched initialization
            n_nodes: Actual number of nodes per sample
            bb_pad: Number of padding dims in BB one-hot
            rxn_pad: Number of padding dims in rxn one-hot
            device: Target device for tensors
        """
        bb_onehot, rxn_onehot, node_mask = create_masked_graph(
            max_nodes=max_nodes,
            vocab_num_bbs=cls.VOCAB_NUM_BBS,
            vocab_num_rxns=cls.VOCAB_NUM_RXNS,
            vocab_num_centers=cls.VOCAB_NUM_CENTERS,
            bb_pad=bb_pad,
            rxn_pad=rxn_pad,
            batch_size=batch_size,
            n_nodes=n_nodes,
            device=device,
        )
        return cls(bb_onehot, rxn_onehot, node_mask, bb_pad, rxn_pad)

    # ============ Core Properties with Cache Invalidation ============

    @property
    def bb_onehot(self) -> torch.Tensor:
        """Building block one-hot tensor. Setting this invalidates BB caches."""
        return self._bb_onehot

    @bb_onehot.setter
    def bb_onehot(self, value: torch.Tensor):
        self._bb_onehot = value
        self._invalidate_bb_caches()

    @property
    def rxn_onehot(self) -> torch.Tensor:
        """Reaction one-hot tensor. Setting this invalidates reaction caches."""
        return self._rxn_onehot

    @rxn_onehot.setter
    def rxn_onehot(self, value: torch.Tensor):
        self._rxn_onehot = value
        self._invalidate_rxn_caches()

    @property
    def node_mask(self) -> torch.Tensor:
        """Node validity mask."""
        return self._node_mask

    @node_mask.setter
    def node_mask(self, value: torch.Tensor):
        self._node_mask = value

    @property
    def is_batched(self) -> bool:
        """Whether this graph is batched."""
        return self._is_batched

    @property
    def device(self) -> torch.device:
        """Device of this graph (from primary tensor)."""
        return self._bb_onehot.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of this graph (from primary tensor)."""
        return self._bb_onehot.dtype

    # ============ Derived Properties (Lazy, Cached) ============

    @property
    def bb_indices(self) -> torch.Tensor:
        """Building block indices (N,) or (B, N). Computed lazily."""
        if self._bb_indices is None:
            self._bb_indices = bb_onehot_to_indices(self._bb_onehot)
        return self._bb_indices

    @property
    def rxn_indices(self) -> torch.Tensor:
        """Flat reaction indices (N, N) or (B, N, N). Computed lazily."""
        if self._rxn_indices is None:
            self._rxn_indices = rxn_onehot_to_indices(self._rxn_onehot)
        return self._rxn_indices

    @property
    def rxn_tuple(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reaction tuples (E, 5) or List[(E_i, 5)]. Computed lazily."""
        if self._rxn_tuple is None:
            self._rxn_tuple = rxn_onehot_to_tuple(
                self._rxn_onehot, self.VOCAB_NUM_CENTERS
            )
        return self._rxn_tuple

    @property
    def building_blocks(self) -> Union[List[BuildingBlock], List[List[BuildingBlock]]]:
        """BuildingBlock objects. Computed lazily.

        Returns:
            list[BuildingBlock] if unbatched, list[list[BuildingBlock]] if batched.
        """
        if self._building_blocks is None:
            if self.is_batched:
                self._building_blocks = [
                    [BuildingBlock(int(idx)) for idx in self.bb_indices[b]]
                    for b in range(self.batch_size)
                ]
            else:
                self._building_blocks = [
                    BuildingBlock(int(idx)) for idx in self.bb_indices
                ]
        return self._building_blocks

    @property
    def reactions(self) -> Union[List[Reaction], List[List[Reaction]]]:
        """Reaction objects. Computed lazily.

        Returns:
            list[Reaction] if unbatched, list[list[Reaction]] if batched.
        """
        if self._reactions is None:
            if self.is_batched:
                self._reactions = [
                    [Reaction(rxn) for rxn in self.rxn_tuple[b]]
                    for b in range(self.batch_size)
                ]
            else:
                self._reactions = [Reaction(rxn) for rxn in self.rxn_tuple]
        return self._reactions

    # ============ Cache Invalidation ============

    def _invalidate_bb_caches(self):
        """Invalidate caches that depend on bb_onehot."""
        self._bb_indices = None
        self._building_blocks = None

    def _invalidate_rxn_caches(self):
        """Invalidate caches that depend on rxn_onehot."""
        self._rxn_indices = None
        self._rxn_tuple = None
        self._reactions = None

    # ============ Dimension Properties ============

    @property
    def bb_onehot_dim(self) -> int:
        """Dimension of BB one-hot."""
        return self._bb_onehot.shape[-1]

    @property
    def rxn_onehot_dim(self) -> int:
        """Dimension of reaction one-hot."""
        return self._rxn_onehot.shape[-1]

    @property
    def batch_size(self) -> int:
        """Batch size (raises if not batched)."""
        if not self.is_batched:
            raise ValueError("batch_size is only valid for batched BBRxnGraph")
        return self._bb_onehot.shape[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes (max nodes if batched)."""
        return self._bb_onehot.shape[-2]

    @property
    def lengths(self) -> torch.Tensor:
        """Actual number of valid nodes per sample."""
        if self.is_batched:
            return self._node_mask.sum(dim=1).long()
        return self._node_mask.sum().long().unsqueeze(0)

    # ============ Collection Interface ============

    def __len__(self) -> int:
        """Number of graphs in batch."""
        if not self.is_batched:
            raise TypeError("len() is only valid for batched BBRxnGraph")
        return self.batch_size

    def __getitem__(self, idx: int) -> "BBRxnGraph":
        """Get single graph from batch."""
        if not self.is_batched:
            raise TypeError("Indexing is only valid for batched BBRxnGraph")

        n = int(self.lengths[idx].item())
        return BBRxnGraph(
            bb_onehot=self._bb_onehot[idx, :n],
            rxn_onehot=self._rxn_onehot[idx, :n, :n],
            node_mask=self._node_mask[idx, :n] if self._node_mask is not None else None,
            bb_pad=self.bb_pad,
            rxn_pad=self.rxn_pad,
        )

    # ============ Derived Masks ============

    @property
    def unmasked_bbs(self) -> torch.Tensor:
        """Boolean tensor indicating which BBs are unmasked (have a real building block)."""
        return self._bb_onehot[..., -1] == 0

    @property
    def unmasked_rxns(self) -> torch.Tensor:
        """Boolean tensor indicating which reactions are not MASK.

        Note: With padding as MASK (matching old code), we only check the MASK channel.
        """
        return self._rxn_onehot[..., -1] == 0

    @property
    def compatibility_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility masks for nodes and edges.

        Returns:
            (bb_mask, rxn_mask) boolean tensors of same shapes as one-hots
        """
        return compute_compatibility_masks(
            self._bb_onehot, self._rxn_onehot, COMPATIBILITY
        )

    @property
    def ground_truth_atom_mask(self) -> torch.Tensor:
        """Ground-truth valid-atom mask (True = valid, False = removed/padded).

        Shape: [n_nodes, MAX_ATOMS_PER_BB] for unbatched, or batched accordingly.
        """
        if self.is_batched:
            n_bbs = self._bb_onehot.shape[1]
            mask = torch.ones(
                self.batch_size,
                n_bbs,
                MAX_ATOMS_PER_BB,
                dtype=torch.bool,
                device=self.device,
            )

            for b in range(self.batch_size):
                for reaction in self.reactions[b]:
                    node1_order = reaction.node1_idx
                    node2_order = reaction.node2_idx
                    center1_idx = reaction.center1_idx
                    center2_idx = reaction.center2_idx
                    if reaction.r1_atom_dropped:
                        r1_atom_idx = self.building_blocks[b][node1_order].get_atom_idx(
                            center1_idx
                        )
                        mask[b, node1_order, r1_atom_idx] = False
                    if reaction.r2_atom_dropped:
                        r2_atom_idx = self.building_blocks[b][node2_order].get_atom_idx(
                            center2_idx
                        )
                        mask[b, node2_order, r2_atom_idx] = False

                for i in range(n_bbs):
                    n_atoms = self.building_blocks[b][i].num_atoms
                    mask[b, i, n_atoms:] = False

                pad_nodes = self._node_mask[b].to(dtype=torch.bool) == 0
                if pad_nodes.any():
                    mask[b, pad_nodes] = False

            flat = mask.reshape(self.batch_size, -1)
            return flat.long()

        # Unbatched
        n_bbs = self._bb_onehot.shape[0]
        mask = torch.ones(n_bbs, MAX_ATOMS_PER_BB, dtype=torch.bool, device=self.device)

        for reaction in self.reactions:
            node1_order, node2_order = reaction.node1_idx, reaction.node2_idx
            center1_idx, center2_idx = reaction.center1_idx, reaction.center2_idx
            if reaction.r1_atom_dropped:
                r1_atom_idx = self.building_blocks[node1_order].get_atom_idx(
                    center1_idx
                )
                mask[node1_order, r1_atom_idx] = False
            if reaction.r2_atom_dropped:
                r2_atom_idx = self.building_blocks[node2_order].get_atom_idx(
                    center2_idx
                )
                mask[node2_order, r2_atom_idx] = False

        for i in range(n_bbs):
            n_atoms = self.building_blocks[i].num_atoms
            mask[i, n_atoms:] = False

        pad_nodes = self._node_mask.to(dtype=torch.bool) == 0
        if pad_nodes.any():
            mask[pad_nodes] = False

        flat = mask.reshape(-1)
        return flat.long()

    @property
    def partial_atom_mask(self) -> torch.Tensor:
        """Partial validity mask for atoms per building block.

        True indicates valid atom positions for current graph state.
        - For non-mask building blocks: exactly num_atoms are True
        - For mask building blocks: all MAX_ATOMS_PER_BB positions are True

        Shape: [n_nodes, MAX_ATOMS_PER_BB] for unbatched, or batched accordingly.
        """
        if self.is_batched:
            n_bbs = self._bb_onehot.shape[1]
            valid = torch.zeros(
                self.batch_size,
                n_bbs,
                MAX_ATOMS_PER_BB,
                dtype=torch.bool,
                device=self.device,
            )
            for b in range(self.batch_size):
                for i in range(n_bbs):
                    bb = self.building_blocks[b][i]
                    n_valid = MAX_ATOMS_PER_BB if bb.is_mask else bb.num_atoms
                    if n_valid > 0:
                        valid[b, i, :n_valid] = True
                pad_nodes = self._node_mask[b].to(dtype=torch.bool) == 0
                if pad_nodes.any():
                    valid[b, pad_nodes] = False
            return valid.reshape(self.batch_size, -1).long()

        n_bbs = self._bb_onehot.shape[0]
        valid = torch.zeros(
            n_bbs, MAX_ATOMS_PER_BB, dtype=torch.bool, device=self.device
        )
        for i in range(n_bbs):
            bb = self.building_blocks[i]
            n_valid = MAX_ATOMS_PER_BB if bb.is_mask else bb.num_atoms
            if n_valid > 0:
                valid[i, :n_valid] = True
        pad_nodes = self._node_mask.to(dtype=torch.bool) == 0
        if pad_nodes.any():
            valid[pad_nodes] = False
        return valid.reshape(-1).long()

    # ============ RDKit Conversion ============

    def build_rdkit(self, return_smiles: bool = False):
        """Build RDKit molecules/SMILES from this graph.

        Args:
            return_smiles: If True, return SMILES strings instead of Mol objects

        Returns:
            Molecule/SMILES or None for single graph, list for batched.

        Raises:
            AssertionError: If any node or reaction is still masked.
        """
        from syncogen.utils.rdkit import build_molecule

        if not self.is_batched:
            n = int(self.lengths.item())
            assert (
                self.unmasked_bbs[:n].all() and self.unmasked_rxns[:n, :n].all()
            ), "Cannot build RDKit molecules with masked building blocks or reactions"
            return build_molecule(
                self.bb_indices[:n], self.rxn_tuple, smiles=return_smiles
            )

        outputs = []
        for b in range(self.batch_size):
            n = int(self.lengths[b].item())
            assert (
                self.unmasked_bbs[b, :n].all() and self.unmasked_rxns[b, :n, :n].all()
            ), f"Cannot build RDKit molecules with masked building blocks or reactions (batch {b})"
            outputs.append(
                build_molecule(
                    self.bb_indices[b, :n], self.rxn_tuple[b], smiles=return_smiles
                )
            )
        return outputs

    # ============ Utility Methods ============

    def to(self, device: torch.device) -> "BBRxnGraph":
        """Move graph to device."""
        return BBRxnGraph(
            bb_onehot=self._bb_onehot.to(device),
            rxn_onehot=self._rxn_onehot.to(device),
            node_mask=(
                self._node_mask.to(device) if self._node_mask is not None else None
            ),
            bb_pad=self.bb_pad,
            rxn_pad=self.rxn_pad,
        )

    def clone(self) -> "BBRxnGraph":
        """Create a deep copy of this graph."""
        return BBRxnGraph(
            bb_onehot=self._bb_onehot.clone(),
            rxn_onehot=self._rxn_onehot.clone(),
            node_mask=self._node_mask.clone() if self._node_mask is not None else None,
            bb_pad=self.bb_pad,
            rxn_pad=self.rxn_pad,
        )

    def apply_edge_givens(self) -> "BBRxnGraph":
        """Apply edge invariants: diagonals -> NO-EDGE, padding -> zeros.

        Call this after any operation that might violate edge invariants
        (e.g., noising, sampling steps).

        Returns:
            Self (modified in-place) for chaining.
        """
        self._rxn_onehot = apply_edge_givens(self._rxn_onehot, self._node_mask)
        self._invalidate_rxn_caches()
        return self

    def __repr__(self) -> str:
        if self.is_batched:
            return (
                f"BBRxnGraph(batched, B={self.batch_size}, max_nodes={self.num_nodes})"
            )
        return f"BBRxnGraph(nodes={self.num_nodes})"

    ##########################################################################################
    # The following function is only used in the backbone and during reconstruction.
    ##########################################################################################

    def calculate_bonds(self, reindex: bool = True, as_onehot_adj_tensor: bool = False):
        """Get molecular bonds from graph structure.

        Computes bonds for the final molecule by:
        1. Getting intra-fragment bonds from each building block
        2. Adding inter-fragment bonds from reactions (accounting for dropped atoms)

        Args:
            reindex: Whether to reindex atoms to contiguous indices starting from 0
            as_onehot_adj_tensor: Whether to return as one-hot adjacency tensor

        Returns:
            Either:
            - List of tuples (atom1_idx, atom2_idx, bond_type)
            - Tensor [n_atoms, n_atoms, 5] with one-hot bond types:
              [single, double, triple, aromatic, masked]

        Note:
            Only supports unbatched graphs. For batched graphs, index into
            individual graphs first.
        """
        from rdkit import Chem

        if self.is_batched:
            raise ValueError(
                "calculate_bonds only supports unbatched graphs. Use graph[i].calculate_bonds()"
            )

        # Get the atom mask to know which atoms are valid (not dropped/padded)
        atom_mask = self.ground_truth_atom_mask
        flat_mask = atom_mask.reshape(-1).bool()  # True = valid atom

        bonds = []
        n_bbs = len(self.building_blocks)

        # 1. Collect intra-fragment bonds from each building block
        for bb_idx, bb in enumerate(self.building_blocks):
            if bb.is_mask:
                continue

            global_offset = bb_idx * MAX_ATOMS_PER_BB

            for bond in bb.mol.GetBonds():
                begin_local = bond.GetBeginAtomIdx()
                end_local = bond.GetEndAtomIdx()
                begin_global = begin_local + global_offset
                end_global = end_local + global_offset

                # Only include bond if both atoms are valid (not dropped)
                if flat_mask[begin_global] and flat_mask[end_global]:
                    bonds.append((begin_global, end_global, bond.GetBondType()))

        # 2. Add inter-fragment bonds from reactions
        for reaction in self.reactions:
            if reaction.is_mask:
                continue

            bb1 = self.building_blocks[reaction.node1_idx]
            bb2 = self.building_blocks[reaction.node2_idx]

            # Get the atom indices at the reaction centers
            center1_atom_local = bb1.get_atom_idx(reaction.center1_idx)
            center2_atom_local = bb2.get_atom_idx(reaction.center2_idx)

            # Convert to global indices
            atom1_global = center1_atom_local + reaction.node1_idx * MAX_ATOMS_PER_BB
            atom2_global = center2_atom_local + reaction.node2_idx * MAX_ATOMS_PER_BB

            # If an atom is dropped, use its neighbor instead
            if reaction.r1_atom_dropped:
                atom = bb1.mol.GetAtomWithIdx(center1_atom_local)
                neighbor = atom.GetNeighbors()[
                    0
                ]  # First neighbor (leaving group removed)
                atom1_global = neighbor.GetIdx() + reaction.node1_idx * MAX_ATOMS_PER_BB

            if reaction.r2_atom_dropped:
                atom = bb2.mol.GetAtomWithIdx(center2_atom_local)
                neighbor = atom.GetNeighbors()[0]
                atom2_global = neighbor.GetIdx() + reaction.node2_idx * MAX_ATOMS_PER_BB

            # Reaction bonds are single bonds
            bonds.append((atom1_global, atom2_global, Chem.BondType.SINGLE))

        # 3. Optionally reindex to contiguous atom indices
        if reindex:
            atoms = set()
            for a, b, _ in bonds:
                atoms.add(a)
                atoms.add(b)

            sorted_atoms = sorted(atoms)
            atom_mapping = {old: new for new, old in enumerate(sorted_atoms)}
            bonds = [(atom_mapping[a], atom_mapping[b], bt) for a, b, bt in bonds]

        # 4. Optionally convert to one-hot adjacency tensor
        if as_onehot_adj_tensor:
            total_atoms = n_bbs * MAX_ATOMS_PER_BB
            adj = torch.zeros((total_atoms, total_atoms, 5), device=self.device)

            for a1, a2, bond_type in bonds:
                if bond_type == Chem.BondType.SINGLE:
                    idx = 0
                elif bond_type == Chem.BondType.DOUBLE:
                    idx = 1
                elif bond_type == Chem.BondType.TRIPLE:
                    idx = 2
                elif bond_type == Chem.BondType.AROMATIC:
                    idx = 3
                else:
                    continue  # Skip unknown bond types

                adj[a1, a2, idx] = 1
                adj[a2, a1, idx] = 1

            return adj

        return bonds
