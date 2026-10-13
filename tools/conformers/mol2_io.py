"""MOL2 writing for SynCoGen conformer datasets, plus a round-trip verifier.

The training loader (syncogen/utils/file_readers.py + syncogen/data/dataloader.py)
reads conformers out of an LMDB whose values are MOL2 text and whose keys are
``mol_{data_index}_final_conf_{j}``. Nothing in the released repo WRITES that
format, so this module reproduces it.

The one invariant that makes or breaks training
-----------------------------------------------
``SyncogenDataset.get`` builds the per-fragment atom mask

    atom_mask = BBRxnGraph.from_indices(...).ground_truth_atom_mask
                -> [n_fragments, MAX_ATOMS_PER_BB], True = real atom

where position ``a`` of fragment ``f`` is building block ``f``'s own SMILES atom
index ``a``; ``mask[f, a]`` is False when atom ``a`` was the leaving group of a
reaction, and False for ``a >= bb.num_atoms``. ``mol2_to_coordinates`` then walks
the True positions of each fragment in ascending order and fills them with that
fragment's atoms **in the order they appear in the MOL2 file**.

So the i-th valid mask slot must hold the i-th atom that ``graph.build_rdkit()``
emits for that fragment. ``build_rdkit`` -> ``RDKitMoleculeAssembly.to_mol()``
iterates its atom graph in insertion order, which is fragment 0's atoms in
ascending BB-SMILES index, then fragment 1's, and so on, with leaving-group atoms
absent. Writing atoms in ``mol.GetAtoms()`` order therefore satisfies the
invariant by construction -- and ``verify_roundtrip`` below asserts it rather
than trusting it.

The same correspondence is what ``build_molecules_from_graphs`` relies on in the
opposite direction at sampling time (``valid_coords = atom_coords[atom_mask]``
handed to ``set_mol_coordinates`` in mol atom order), so matching it here keeps
training and sampling consistent.

Conventions below were read off the published SynSpace LMDB
(DreiSSB/SynSpace, mol2s.lmdb), not guessed:
  * atom name  = ``{Element}_{bb_idx}_{frag_order}``, on hydrogens too, where a
    hydrogen inherits bb_idx/frag_order from the heavy atom it hangs off
  * heavy atoms first grouped by ascending frag_order, then all hydrogens
    (exactly RDKit ``AddHs`` ordering)
  * atom type column is the plain element symbol, not a Sybyl type
  * bond types ``1`` single, ``2`` double, ``3`` triple, ``4`` aromatic
  * ``subst_id`` 1 and ``subst_name`` = the record key, charges all 0.0000
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem

BOND_SYMBOL = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "4",
}


def annotate_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Copy frag_order/global_frag_id from each heavy atom onto its hydrogens.

    ``to_mol`` sets these int props on heavy atoms only; ``AddHs`` creates
    hydrogens without them. SynSpace names hydrogens after their parent
    fragment, so propagate.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("frag_order"):
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy:
            raise ValueError(
                f"atom {atom.GetIdx()} ({atom.GetSymbol()}) has no annotated heavy neighbour"
            )
        parent = heavy[0]
        atom.SetIntProp("frag_order", parent.GetIntProp("frag_order"))
        atom.SetIntProp("global_frag_id", parent.GetIntProp("global_frag_id"))
    return mol


def write_mol2(mol: Chem.Mol, key: str, conf_id: int = 0) -> str:
    """Render one conformer as MOL2 text in the layout SynSpace uses.

    Atoms are emitted in ``mol.GetAtoms()`` order, which is what the loader's
    fragment/atom correspondence depends on. Every atom must already carry
    ``frag_order`` and ``global_frag_id`` (see ``annotate_hydrogens``).
    """
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    out: List[str] = [
        "@<TRIPOS>MOLECULE",
        key,
        f"{n_atoms} {n_bonds} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        pos = conf.GetAtomPosition(i)
        sym = atom.GetSymbol()
        name = (
            f"{sym}_{atom.GetIntProp('global_frag_id')}_{atom.GetIntProp('frag_order')}"
        )
        out.append(
            f"{i + 1:>7} {name:<11}"
            f"{pos.x:>9.4f} {pos.y:>9.4f} {pos.z:>9.4f} "
            f"{sym:<6} 1 {key}  0.0000"
        )

    out.append("@<TRIPOS>BOND")
    for bond in mol.GetBonds():
        bt = BOND_SYMBOL.get(bond.GetBondType())
        if bt is None:
            raise ValueError(f"unsupported bond type {bond.GetBondType()}")
        out.append(
            f"{bond.GetIdx() + 1:>6}{bond.GetBeginAtomIdx() + 1:>6}"
            f"{bond.GetEndAtomIdx() + 1:>6} {bt}"
        )

    out.append("")
    return "\n".join(out)


def verify_roundtrip(
    mol2_text: str,
    mol: Chem.Mol,
    atom_mask: torch.Tensor,
    conf_id: int = 0,
    tol: float = 5e-4,
) -> None:
    """Assert the repo's own reader recovers our coordinates exactly.

    Parses ``mol2_text`` with the production reader and checks that
      * the recovered valid-slot pattern equals ``atom_mask`` exactly, and
      * flattening the recovered coordinates by that mask reproduces the heavy
        atoms of ``mol`` in mol order -- the same operation
        ``build_molecules_from_graphs`` performs at sampling time.

    ``tol`` is just the MOL2 text's 4-decimal rounding.
    """
    from syncogen.utils.file_readers import mol2_to_coordinates

    coords, coords_mask, _ = mol2_to_coordinates(
        mol2_text.encode(), atom_mask=atom_mask
    )

    if coords_mask.shape != atom_mask.shape:
        raise AssertionError(
            f"shape mismatch: reader {tuple(coords_mask.shape)} vs mask {tuple(atom_mask.shape)}"
        )
    if not torch.equal(coords_mask, atom_mask.bool()):
        raise AssertionError(
            "recovered valid-atom pattern differs from ground_truth_atom_mask "
            f"({int(coords_mask.sum())} vs {int(atom_mask.sum())} valid slots)"
        )

    # The sampling-time inverse: i-th valid slot <-> i-th heavy atom of mol.
    recovered = coords.reshape(-1, 3)[atom_mask.bool().reshape(-1)]

    conf = mol.GetConformer(conf_id)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) != recovered.shape[0]:
        raise AssertionError(
            f"heavy atom count {len(heavy)} != recovered {recovered.shape[0]}"
        )
    expected = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in heavy], dtype=torch.float32
    )

    dev = (recovered - expected).abs().max().item()
    if dev > tol:
        raise AssertionError(f"coordinate round-trip off by {dev:.6f} A (tol {tol})")


def graph_from_data(data) -> Tuple[object, Chem.Mol, torch.Tensor]:
    """Rebuild (BBRxnGraph, heavy-atom Mol, atom_mask) from a dataset Data object.

    Mirrors ``SyncogenDataset.get`` so the mask we validate against is byte-for-byte
    the one training will compute.
    """
    from syncogen.api.graph.graph import BBRxnGraph
    from syncogen.constants.constants import MAX_ATOMS_PER_BB
    from syncogen.data.dataloader import expand_edge_indices_to_matrix

    n_nodes = data.x.shape[0]
    node_mask = torch.ones(n_nodes, dtype=torch.bool, device=data.x.device)
    rxn_matrix = expand_edge_indices_to_matrix(
        data.edge_index, data.edge_attr, n_nodes, data.x.device
    )
    graph = BBRxnGraph.from_indices(data.x, rxn_matrix, node_mask=node_mask)
    atom_mask = graph.ground_truth_atom_mask.bool().reshape(n_nodes, MAX_ATOMS_PER_BB)
    mol = graph.build_rdkit(return_smiles=False)
    return graph, mol, atom_mask
