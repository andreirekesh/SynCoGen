"""
File readers for molecular structure files with building block annotations.
"""

from numpy import ndarray
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import torch
import numpy as np
import lmdb
import random
from syncogen.constants.constants import MAX_ATOMS_PER_BB
import pickle


def get_mol2_data(key: str, lmdb_path: Path) -> bytes:
    """Get MOL2 data from LMDB database.

    Args:
        key: Key to look up in LMDB (e.g. "mol_0_final_conf_0")
        lmdb_path: Path to LMDB database

    Returns:
        mol2_data: MOL2 file contents as bytes
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            mol2_data = txn.get(key.encode())
            if mol2_data is None:
                raise KeyError(f"Key {key} not found in LMDB at {lmdb_path}")
            return mol2_data
    finally:
        env.close()


def get_conformer_keys(data_index: Union[str, int], lmdb_path: Path) -> List[str]:
    """Get list of available conformer keys for a molecule.

    Args:
        data_index: Base identifier for the molecule (e.g. "mol_0")
        lmdb_path: Path to LMDB database

    Returns:
        conformer_keys: List of valid conformer keys for this molecule
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            conformer_keys = []
            i = 0
            while True:
                key = f"mol_{data_index}_final_conf_{i}"
                if txn.get(key.encode()) is None:
                    break
                conformer_keys.append(key)
                i += 1
            return conformer_keys
    finally:
        env.close()


def select_conformer_key(
    data_index: Union[str, int], lmdb_path: Path, random_conformer: bool = False
) -> str:
    """Select a conformer key for a molecule.

    Args:
        data_index: Base identifier for the molecule (e.g. "mol_0")
        lmdb_path: Path to LMDB database
        random_conformer: If True, randomly select a conformer, otherwise use conformer 0

    Returns:
        key: Selected conformer key (e.g. "mol_0_final_conf_0")
    """
    conformer_keys = get_conformer_keys(data_index, lmdb_path)

    if len(conformer_keys) == 0:
        raise KeyError(f"No conformers found for {data_index} in {lmdb_path}")

    return random.choice(conformer_keys) if random_conformer else conformer_keys[0]


def parse_mol2_file(
    mol2_data: bytes,
) -> Tuple[np.ndarray, List[Tuple[int, int, str]], Dict[int, Tuple[str, int, int]]]:
    """Parse MOL2 data to extract coordinates, bonds, and building block annotations.

    Args:
        mol2_data: MOL2 file contents as bytes

    Returns:
        coords: numpy array of shape [n_atoms, 3] with atomic coordinates
        bonds: List of (atom1_idx, atom2_idx, bond_type) tuples (0-indexed)
        annotations: Dict mapping atom index to (element, bb_idx, order_idx)
    """
    lines = mol2_data.decode("utf-8").split("\n")

    # Find sections
    atom_start = None
    bond_start = None

    for i, line in enumerate(lines):
        if line.strip() == "@<TRIPOS>ATOM":
            atom_start = i + 1
        elif line.strip() == "@<TRIPOS>BOND":
            bond_start = i + 1
            break

    if atom_start is None:
        raise ValueError("No @<TRIPOS>ATOM section found in MOL2 data")

    # Parse atoms
    coords = []
    annotations = {}

    for i in range(atom_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("@<TRIPOS>"):
            break

        parts = line.split()
        if len(parts) < 6:
            continue

        # MOL2 format: atom_id atom_name x y z atom_type ...
        atom_idx = int(parts[0]) - 1  # Convert to 0-indexed
        atom_name = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

        coords.append([x, y, z])

        # Parse building block annotation from atom name (e.g., "C_44_0")
        if "_" in atom_name:
            parts_name = atom_name.split("_")
            if len(parts_name) >= 3:
                element = parts_name[0]
                try:
                    bb_idx = int(parts_name[1])
                    order_idx = int(parts_name[2])
                    annotations[atom_idx] = (element, bb_idx, order_idx)
                except ValueError:
                    pass

    coords = np.array(coords)

    # Parse bonds
    bonds = []

    if bond_start is not None:
        for i in range(bond_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith("@<TRIPOS>"):
                break

            parts = line.split()
            if len(parts) < 4:
                continue

            # MOL2 format: bond_id atom1 atom2 bond_type
            atom1 = int(parts[1]) - 1  # Convert to 0-indexed
            atom2 = int(parts[2]) - 1
            bond_type = parts[3]

            bonds.append((atom1, atom2, bond_type))

    return coords, bonds, annotations


def mol2_to_coordinates(
    mol2_data: bytes,
    mask_value: float = 0.0,
    return_bonds: bool = False,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Parse MOL2 data into fragment-grouped coordinates, accounting for dropped atoms.

    Args:
        mol2_data: MOL2 file contents with building block annotations
        mask_value: Value to use for masked atoms
        return_bonds: If True, also return bonds mapped to flattened padded indices
        atom_mask: Optional tensor of shape [n_fragments, MAX_ATOMS] indicating valid atom positions.
                   If provided, atoms are placed skipping over invalid positions (dropped atoms).
                   If None, atoms are placed sequentially (old behavior, incorrect for reactions).

    Returns:
        coords_tensor: torch.Tensor of shape [n_fragments, MAX_ATOMS, 3]
        coords_mask: torch.Tensor of shape [n_fragments, MAX_ATOMS]
        bonds: Optional torch.Tensor of shape [n_bonds, 3] containing (flat_i, flat_j, bond_type)
    """
    # Parse MOL2 data
    coords, bonds, annotations = parse_mol2_file(mol2_data)

    # Determine number of fragments from annotations
    order_indices = [order_idx for (_, _, order_idx) in annotations.values()]
    n_fragments = max(order_indices) + 1 if order_indices else 0

    coords_tensor = torch.full((n_fragments, MAX_ATOMS_PER_BB, 3), mask_value)
    coords_mask = torch.zeros((n_fragments, MAX_ATOMS_PER_BB), dtype=torch.bool)

    # Group atoms by fragment (order_idx)
    frag_atoms: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for atom_idx, (element, bb_idx, order_idx) in annotations.items():
        if element == "H":  # Skip hydrogens
            continue
        if order_idx not in frag_atoms:
            frag_atoms[order_idx] = []
        frag_atoms[order_idx].append((atom_idx, coords[atom_idx]))

    # Fill coordinates tensor
    # Keep a per-fragment mapping: global atom index -> local position index
    frag_atoms_global: Dict[int, Dict[int, int]] = (
        {}
    )  # order_idx -> {global_idx: local_position}
    for order_idx in sorted(frag_atoms.keys()):
        atoms = frag_atoms[order_idx]
        frag_atoms_global[order_idx] = {}

        if atom_mask is not None and order_idx < atom_mask.shape[0]:
            # Use atom_mask to skip over invalid positions (dropped atoms)
            valid_positions = atom_mask[order_idx].bool()  # True = valid position
            position_idx = 0  # Index into valid positions
            for global_idx, coord in atoms:
                # Find next valid position
                while (
                    position_idx < MAX_ATOMS_PER_BB
                    and not valid_positions[position_idx]
                ):
                    position_idx += 1

                if position_idx < MAX_ATOMS_PER_BB:
                    coords_tensor[order_idx, position_idx] = torch.tensor(
                        coord, dtype=torch.float32
                    )
                    coords_mask[order_idx, position_idx] = True
                    frag_atoms_global[order_idx][
                        global_idx
                    ] = position_idx  # Store actual position
                    position_idx += 1
        else:
            # Old behavior: place atoms sequentially (incorrect for reactions with dropped atoms)
            for local_idx, (global_idx, coord) in enumerate(atoms):
                if local_idx < MAX_ATOMS_PER_BB:
                    coords_tensor[order_idx, local_idx] = torch.tensor(
                        coord, dtype=torch.float32
                    )
                    coords_mask[order_idx, local_idx] = True
                    frag_atoms_global[order_idx][
                        global_idx
                    ] = local_idx  # Store actual position

    if not return_bonds:
        return coords_tensor, coords_mask, None

    # Build mapping: global atom index -> flattened padded index (order_idx * MAX_ATOMS_PER_BB + local_idx)
    global_to_flat: Dict[int, int] = {}
    for order_idx in sorted(frag_atoms_global.keys()):
        for global_idx, local_idx in frag_atoms_global[order_idx].items():
            global_to_flat[global_idx] = order_idx * MAX_ATOMS_PER_BB + local_idx

    # Convert bond types to integers (consistent with mol2_to_bonds)
    bond_type_map = {
        "1": 1,
        "SINGLE": 1,
        "single": 1,
        "2": 2,
        "DOUBLE": 2,
        "double": 2,
        "3": 3,
        "TRIPLE": 3,
        "triple": 3,
        "4": 4,
        "ar": 4,
        "AROMATIC": 4,
        "aromatic": 4,
    }
    mapped_bonds: List[List[int]] = []
    for atom1, atom2, bond_type in bonds:
        if atom1 in global_to_flat and atom2 in global_to_flat:
            bt = bond_type_map.get(bond_type, 1)
            mapped_bonds.append([global_to_flat[atom1], global_to_flat[atom2], bt])

    bonds_tensor = (
        torch.tensor(mapped_bonds, dtype=torch.long)
        if mapped_bonds
        else torch.zeros((0, 3), dtype=torch.long)
    )
    return coords_tensor, coords_mask, bonds_tensor


def mol2_to_bonds(mol2_data: bytes) -> torch.Tensor:
    """Parse MOL2 data to extract bond information.

    Args:
        mol2_data: MOL2 file contents

    Returns:
        bonds: torch.Tensor of shape [n_bonds, 3] containing (atom1, atom2, bond_type)
               Bond types: 1=single, 2=double, 3=triple, 4=aromatic
    """
    _, bonds, _ = parse_mol2_file(mol2_data)

    # Convert bond types to integers
    bond_type_map = {
        "1": 1,
        "SINGLE": 1,
        "single": 1,
        "2": 2,
        "DOUBLE": 2,
        "double": 2,
        "3": 3,
        "TRIPLE": 3,
        "triple": 3,
        "4": 4,
        "ar": 4,
        "AROMATIC": 4,
        "aromatic": 4,
    }

    bonds_tensor = []
    for atom1, atom2, bond_type in bonds:
        bt = bond_type_map.get(bond_type, 1)  # Default to single
        bonds_tensor.append([atom1, atom2, bt])

    return torch.tensor(bonds_tensor, dtype=torch.long)


def get_coordinates(
    key: str,
    lmdb_path: Path,
    filetype: str = "mol2",
    mask_value: float = 0.0,
    return_bonds: bool = False,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Read molecular coordinates and optionally bonds from MOL2 data.

    Args:
        key: Conformer key (e.g. "mol_0_final_conf_0")
        lmdb_path: Path to LMDB database
        filetype: File format ('mol2')
        mask_value: Value to use for masked atoms
        return_bonds: If True, also return bond information
        atom_mask: Optional tensor of shape [n_fragments, MAX_ATOMS] indicating valid positions.
                   If provided, atoms are placed skipping over invalid positions.

    Returns:
        coords_tensor: torch.Tensor of shape [n_fragments, max_atoms_per_fragment, 3]
        coords_mask: torch.Tensor of shape [n_fragments, max_atoms_per_fragment]
        bonds: Optional torch.Tensor of shape [n_bonds, 3] (if return_bonds=True)
    """
    if filetype != "mol2":
        raise ValueError("Only MOL2 format is supported")

    mol2_data = get_mol2_data(key, lmdb_path)
    coords, coords_mask, bonds = mol2_to_coordinates(
        mol2_data, mask_value, return_bonds=return_bonds, atom_mask=atom_mask
    )
    return coords, coords_mask, bonds


def get_pharmacophores(
    key: Union[str, int],
    lmdb_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read raw pharmacophore data from LMDB database.

    Args:
        key: LMDB key to look up
        lmdb_path: Path to LMDB database

    Returns:
        types: Raw tensor of pharmacophore type indices
        positions: Raw tensor of 3D coordinates for each pharmacophore
        vectors: Raw tensor of pharmacophore vectors or None if not present
    """
    lmdb_env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        value = txn.get(str(key).encode("utf-8"))
        data = pickle.loads(value)
        types = torch.tensor(data["types"])
        positions = torch.tensor(data["pos"], dtype=torch.float32)
        vectors = torch.tensor(data["vec"]) if data.get("vec") is not None else None

        return types, positions, vectors
