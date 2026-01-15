"""RDKit utility functions for molecule building and validation."""

from pathlib import Path
from typing import Optional, Tuple, List, TYPE_CHECKING

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Geometry import Point3D

from syncogen.api.rdkit.assembly import RDKitMoleculeAssembly
from syncogen.constants.constants import COMPATIBILITY, COORDS_STD

if TYPE_CHECKING:
    from syncogen.api.graph.graph import BBRxnGraph


def is_valid_smiles(smiles_or_mol):
    """Check if a SMILES string is valid and can be fully sanitized."""
    try:
        # Try to create and sanitize the molecule
        if isinstance(smiles_or_mol, str):
            smiles_or_mol = Chem.MolFromSmiles(smiles_or_mol)
        if smiles_or_mol is None:
            return False

        # Ensure full sanitization including kekulization
        Chem.SanitizeMol(smiles_or_mol)
        return True
    except:
        return False


def build_molecule(
    nodes: torch.Tensor, decoded_edges: torch.Tensor, smiles: bool = False
) -> str:
    """Build a molecule from model outputs. Used in calculating pvalid.

    Args:
        nodes: Tensor of node indices [n_nodes]
        decoded_edges: Tensor of decoded model outputs [n_edges, 5] containing
                     (reaction_id, node1_order, node2_order, center1_idx, center2_idx).
                     These are not the same as actions for adding to MFG and must be converted.
        smiles: Whether to return a SMILES string or an RDKit molecule
    Returns:
        SMILES string of the molecule
    """
    try:
        nodes_list = nodes.tolist()
        edges_list = decoded_edges.tolist()
        mfg = RDKitMoleculeAssembly()
        mfg.add_fragment(nodes_list[0])

        # This condition has to be in place because you can sometimes still build a molecule that has too many denoised reactions
        assert (
            len(nodes_list) == len(edges_list) + 1
        ), f"Mismatch in nodes and edges: {len(nodes_list)} != {len(edges_list)} + 1"
        for i, node in enumerate(nodes_list[1:]):
            edge = edges_list[i]
            action = [
                node,
                edge[0],  # reaction_idx
                edge[1],  # node1
                edge[3],  # center1_idx
                edge[4],  # center2_idx
            ]

            if is_valid_action(mfg, action):
                mfg.add_fragment(*action)
            else:
                print("invalid action:", action)
                return None

        return mfg.to_smiles() if smiles else mfg.to_mol()

    except Exception as e:
        print("Failed to build molecule:", e)
        return None


def is_valid_action(mfg: RDKitMoleculeAssembly, action: list) -> bool:
    """Check if an action is valid for the current molecule fragment graph.

    Args:
        mfg: Current molecule fragment graph
        action: List of [new_fragment_global_id, reaction_id, existing_frag_idx, center1_idx, center2_idx]

    Returns:
        bool: Whether the action is valid
    """
    # Unpack action list
    new_frag_global_id = action[0]
    reaction_id = action[1]
    existing_frag_idx = action[2]
    center1_idx = action[3]
    center2_idx = action[4]

    # Check if reaction center is still available on the existing fragment
    if not mfg.fragment_graph.nodes[existing_frag_idx]["rxn_center_available"][
        center1_idx
    ]:
        return False

    # Resolve existing fragment's building block index
    existing_bb_idx = mfg.fragment_graph.nodes[existing_frag_idx]["global_frag_id"]

    # Basic bounds checks for reaction and centers
    if COMPATIBILITY is None:
        return False
    if reaction_id < 0 or reaction_id >= COMPATIBILITY.shape[1]:
        return False

    # Validate using compact compatibility codes:
    #   0 = incompatible, 1 = reactant 1, 2 = reactant 2, 3 = both
    comp_existing = int(COMPATIBILITY[existing_bb_idx, reaction_id, center1_idx].item())
    comp_new = int(COMPATIBILITY[new_frag_global_id, reaction_id, center2_idx].item())

    cond_bb1 = bool(comp_existing & 1)  # existing BB must be valid as reactant 1
    cond_bb2 = bool(comp_new & 2)  # new BB must be valid as reactant 2

    return cond_bb1 and cond_bb2


def build_molecules_from_graphs(
    graphs: "BBRxnGraph",
    coords: Optional[torch.Tensor] = None,
) -> List[Optional[Chem.Mol]]:
    """Build RDKit molecules from graphs with optional coordinates.

    Args:
        graphs: BBRxnGraph object (batched or unbatched)
        coords: Optional [B, N, 3] coordinate tensor

    Returns:
        List of RDKit molecules (None for failed reconstructions).
        Each successful mol has coordinates set if coords was provided.
    """
    n_graphs = graphs.batch_size if graphs.is_batched else 1
    molecules = []

    for i in range(n_graphs):
        graph_i = graphs[i] if graphs.is_batched else graphs

        try:
            mol = graph_i.build_rdkit(return_smiles=False)
            if mol is None:
                print(
                    f"[build_molecules_from_graphs] Failure: build_rdkit returned None for graph index {i}"
                )
                molecules.append(None)
                continue

            # Set coordinates if provided
            if coords is not None:
                try:
                    atom_coords = (
                        coords[i].reshape(-1, 3)
                        if graphs.is_batched
                        else coords.reshape(-1, 3)
                    )
                    atom_mask = graph_i.ground_truth_atom_mask.reshape(-1).bool()
                    valid_coords = atom_coords[: atom_mask.shape[0], :][atom_mask]

                    if valid_coords.shape[0] == mol.GetNumAtoms():
                        mol = set_mol_coordinates(mol, valid_coords.cpu())
                    else:
                        print(
                            f"[build_molecules_from_graphs] Failure: Coordinate/atom mask size mismatch for graph index {i} "
                            f"(valid_coords: {valid_coords.shape[0]}, mol atoms: {mol.GetNumAtoms()})"
                        )
                        # Don't None it, just print warning.
                except Exception as sub_e:
                    print(
                        f"[build_molecules_from_graphs] Exception in setting coordinates for graph index {i}: {sub_e}"
                    )
                    molecules.append(None)
                    continue

            molecules.append(mol)
        except Exception as e:
            print(f"[build_molecules_from_graphs] Exception for graph index {i}: {e}")
            molecules.append(None)

    return molecules


def get_lipinski_descriptors(mols) -> list:
    """Get Lipinski descriptors for a molecule or batch of molecules (Rule of 5).

    Args:
        mols: RDKit molecule or list of RDKit molecules

    Returns:
        list: List of dicts with Lipinski Rule of 5 descriptors (MW, LogP, HBD, HBA)
    """
    # Cast single molecule to list for consistency
    if not isinstance(mols, list):
        mols = [mols]
        single_input = True
    else:
        single_input = False

    results = []
    for mol in mols:
        results.append(
            {
                "MW": Descriptors.MolWt(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
            }
        )

    # Return single dict if input was single molecule
    return results[0] if single_input else results


def calc_energy(mol: Chem.Mol, per_atom: bool = False) -> Optional[float]:
    """Calculate MMFF94 or UFF energy for an RDKit molecule. Prefer MMFF if available, otherwise UFF. Return None if calculation fails."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
            energy = ff.CalcEnergy()
            return energy / mol.GetNumAtoms() if per_atom else energy
        elif AllChem.UFFHasAllMoleculeParams(mol):
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
            energy = ff.CalcEnergy()
            return energy / mol.GetNumAtoms() if per_atom else energy
    except Exception:
        pass
    return None


def save_as_sdf(mol: Chem.Mol, filepath: str, properties: Optional[dict] = None):
    """Save molecule to SDF file with optional properties.

    Args:
        mol: RDKit molecule with conformer
        filepath: Output file path
        properties: Optional dict of properties to add as SDF data fields
    """
    if properties:
        for key, value in properties.items():
            mol.SetProp(str(key), str(value))
    writer = Chem.SDWriter(filepath)
    writer.write(mol)
    writer.close()


def sdf_to_coordinates(sdf_path: str) -> torch.Tensor:
    """Load coordinates from SDF file.

    Args:
        sdf_path: Path to SDF file

    Returns:
        Tensor of shape [n_atoms, 3]
    """
    mol = Chem.MolFromMolFile(sdf_path)
    mol = Chem.RemoveHs(mol)
    return torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)


def mol_to_pharm_cond(
    mol: Chem.Mol,
    batch_size: int,
    n_subset: int,
    center: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract pharmacophore conditioning from a reference molecule.

    Args:
        mol: RDKit molecule with conformer
        batch_size: Number of copies to create (for batched conditioning)
        n_subset: Max pharmacophores per sample (randomly subsampled per batch element)
        center: Center molecule coordinates before extraction
        normalize: Normalize positions by COORDS_STD

    Returns:
        types: [batch_size, n_subset, n_pharm_types] one-hot pharmacophore types
        pos: [batch_size, n_subset, 3] pharmacophore positions
        mask: [batch_size, n_subset] padding mask
    """
    from shepherd_score.extract_profiles import get_pharmacophores
    from syncogen.constants.constants import N_PHARM

    if center:
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )
        centroid = coords.mean(axis=0)
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(*(pos - centroid)))

    types, pos, _ = get_pharmacophores(mol, multi_vector=False)
    if normalize:
        pos = pos / COORDS_STD

    types = torch.tensor(types, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float32)

    # One-hot encoding
    n_classes = N_PHARM + 1  # +1 for padding class
    types_onehot = torch.zeros(len(types), n_classes)
    types_onehot.scatter_(1, types.unsqueeze(1), 1)
    n_total = len(types)

    # Storage for batched outputs
    types_padded = torch.zeros((batch_size, n_subset, n_classes), dtype=torch.float32)
    pos_padded = torch.zeros((batch_size, n_subset, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, n_subset), dtype=torch.float32)
    types_padded[..., -1] = 1  # Default to padding class

    for b in range(batch_size):
        if n_total > n_subset:
            indices = torch.randperm(n_total)[:n_subset]
        else:
            indices = torch.arange(n_total)
        fill_len = min(n_total, n_subset)
        types_padded[b, :fill_len] = types_onehot[indices[:fill_len]]
        pos_padded[b, :fill_len] = pos[indices[:fill_len]]
        mask[b, :fill_len] = 1.0

    return types_padded, pos_padded, mask


def set_mol_coordinates(mol: Chem.Mol, coords: torch.Tensor) -> Chem.Mol:
    """Set 3D coordinates on an RDKit molecule.

    Args:
        mol: RDKit molecule (will be modified in place)
        coords: [n_atoms, 3] coordinate tensor

    Returns:
        Molecule with updated conformer
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(
            i, Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol
