from rdkit import Chem

# Accessed through the module, not `from ... import`, because load_vocabulary()
# REBINDS these globals at runtime; a from-import would capture None forever
# unless the importer happens to run after load_vocabulary().
import syncogen.constants.constants as _constants
from syncogen.constants.constants import (
    BUILDING_BLOCKS_SMI_TO_IDX,
    BUILDING_BLOCKS_IDX_TO_SMI,
    N_BUILDING_BLOCKS,
)


class BuildingBlock:
    """Building block."""

    def __init__(self, smiles_or_mol_or_idx):
        if (
            isinstance(smiles_or_mol_or_idx, int)
            and smiles_or_mol_or_idx == N_BUILDING_BLOCKS
        ):
            self.is_mask = True
            self.idx = N_BUILDING_BLOCKS
            self.smiles = None
            self.mol = None
            self.num_atoms = 0
            return

        self.is_mask = False

        if isinstance(smiles_or_mol_or_idx, str):
            self.smiles = smiles_or_mol_or_idx
            self.idx = BUILDING_BLOCKS_SMI_TO_IDX[self.smiles]["index"]
            self.mol = Chem.MolFromSmiles(self.smiles)
        elif isinstance(smiles_or_mol_or_idx, int):
            self.idx = smiles_or_mol_or_idx
            self.smiles = BUILDING_BLOCKS_IDX_TO_SMI[self.idx]["smiles"]
            self.mol = Chem.MolFromSmiles(self.smiles)
        else:
            self.mol = smiles_or_mol_or_idx
            self.smiles = Chem.MolToSmiles(self.mol)
            self.idx = BUILDING_BLOCKS_SMI_TO_IDX[self.smiles]["index"]

        self.num_atoms = self.mol.GetNumAtoms()

    def get_atom_idx(self, center_idx):
        """BB-local index of reaction centre `center_idx`.

        Under the legacy convention this is the atom that LEAVES (when the reaction
        drops one); under the new "attachment" convention it is the atom that forms
        the new inter-fragment bond and persists. CENTRE_CONVENTION says which.
        """
        if self.is_mask:
            raise ValueError("Cannot get atom index of mask building block")
        return BUILDING_BLOCKS_IDX_TO_SMI[self.idx]["centers"][center_idx]

    def get_leaving_atoms(self, reaction_idx, center_idx):
        """BB-local atom indices removed when this BB reacts at `center_idx`.

        Returns None when the vocabulary predates the multi-atom-drop refactor, so
        callers can fall back to the legacy single-atom behaviour rather than
        silently treating "no leaving atoms" as correct.
        """
        if self.is_mask:
            return []
        lm = _constants.LEAVING_MASK
        if lm is None:
            return None
        bits = int(lm[self.idx, reaction_idx, center_idx])
        if bits == 0:
            return []
        return [a for a in range(_constants.MAX_ATOMS_PER_BB) if (bits >> a) & 1]

    def __str__(self):
        if self.is_mask:
            return "MASK"
        return self.smiles

    def __repr__(self):
        if self.is_mask:
            return "MASK"
        return self.smiles

    def __eq__(self, other: "BuildingBlock"):
        if self.is_mask and other.is_mask:
            return True
        if self.is_mask or other.is_mask:
            return False
        return self.smiles == other.smiles
