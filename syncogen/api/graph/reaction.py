from rdkit import Chem
from typing import Optional, Union, Tuple
from syncogen.constants.constants import REACTIONS, N_REACTIONS


class Reaction:
    """
    Reaction class that can be initialized either from SMARTS pattern or from a 5-tuple
    (reaction_idx, node1_idx, node2_idx, center1_idx, center2_idx).
    """

    def __init__(self, smarts_or_tuple: Union[str, Tuple[int, int, int, int, int]]):
        if isinstance(smarts_or_tuple, str):
            # Initialize from SMARTS
            self.smarts = smarts_or_tuple
            self.reaction_idx = int(REACTIONS[self.smarts]["index"])
            self.node1_idx = None
            self.node2_idx = None
            self.center1_idx = None
            self.center2_idx = None
            self.is_mask = False
            self.is_no_reaction = False
        else:
            # Initialize from 5-tuple
            self.reaction_idx = int(smarts_or_tuple[0])

            # Check for MASK (last index)
            if self.reaction_idx == N_REACTIONS:
                self.is_mask = True
                self.is_no_reaction = False
                self.smarts = None
                self.node1_idx = None
                self.node2_idx = None
                self.center1_idx = None
                self.center2_idx = None
                self.r1_atom_dropped = False
                self.r2_atom_dropped = False
                return

            # Normal reaction
            self.is_mask = False
            self.is_no_reaction = False
            self.node1_idx = int(smarts_or_tuple[1])
            self.node2_idx = int(smarts_or_tuple[2])
            self.center1_idx = int(smarts_or_tuple[3])
            self.center2_idx = int(smarts_or_tuple[4])
            self.smarts = next(k for k, v in REACTIONS.items() if v["index"] == self.reaction_idx)
            self.r1_atom_dropped = bool(REACTIONS[self.smarts]["drop_bools"][0])
            self.r2_atom_dropped = bool(REACTIONS[self.smarts]["drop_bools"][1])

    def __str__(self):
        if self.is_mask:
            return "MASK"
        return self.smarts

    def __repr__(self):
        if self.is_mask:
            return "MASK"
        if self.node1_idx is None:
            return f"Reaction(smarts='{self.smarts}')"
        return f"Reaction(({self.reaction_idx}, {self.node1_idx}, {self.node2_idx}, {self.center1_idx}, {self.center2_idx}))"

    def __eq__(self, other: "Reaction"):
        if self.is_mask and other.is_mask:
            return True
        if self.is_mask or other.is_mask:
            return False
        return (
            self.smarts == other.smarts
            and self.node1_idx == other.node1_idx
            and self.node2_idx == other.node2_idx
            and self.center1_idx == other.center1_idx
            and self.center2_idx == other.center2_idx
        )
