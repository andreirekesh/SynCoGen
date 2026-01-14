import json
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import torch

ROOT_DIR = Path(__file__).parent.parent

_VOCAB: Optional[Dict[str, Any]] = None

BUILDING_BLOCKS = None
BUILDING_BLOCKS_SMI_TO_IDX = None
BUILDING_BLOCKS_IDX_TO_SMI = None
N_BUILDING_BLOCKS = None
MAX_ATOMS_PER_BB = None
FRAGMENT_MACCS = None
FRAGMENT_ATOMFEATS = None
FRAGMENT_BONDFEATS = None
FRAGMENT_ATOMADJ = None
ATOM_TYPES = None
N_ATOM_FEATURES = None
N_BOND_FEATURES = None
REACTIONS = None
N_REACTIONS = None
N_CENTERS = None
COMPATIBILITY = None


def load_vocabulary(vocab_dir: Path) -> Dict[str, Any]:
    """Load vocabulary and update module-level constants."""
    global _VOCAB
    global BUILDING_BLOCKS, BUILDING_BLOCKS_SMI_TO_IDX, BUILDING_BLOCKS_IDX_TO_SMI
    global N_BUILDING_BLOCKS, MAX_ATOMS_PER_BB
    global FRAGMENT_MACCS, FRAGMENT_ATOMFEATS, FRAGMENT_BONDFEATS, FRAGMENT_ATOMADJ
    global ATOM_TYPES, N_ATOM_FEATURES, N_BOND_FEATURES
    global REACTIONS, N_REACTIONS, N_CENTERS, COMPATIBILITY

    vocab_dir = Path(vocab_dir)

    with open(vocab_dir / "building_blocks.json") as f:
        raw_bbs = json.load(f)

    bb_smi_to_idx: Dict[str, Dict[str, Any]] = {}
    bb_idx_to_smi: Dict[int, Dict[str, Any]] = {}
    for idx, (smiles, centers) in enumerate(raw_bbs.items()):
        bb_smi_to_idx[smiles] = {"index": idx, "centers": centers}
        bb_idx_to_smi[idx] = {"smiles": smiles, "centers": centers}

    n_building_blocks = len(bb_smi_to_idx)

    with open(vocab_dir / "reactions.json") as f:
        raw_reactions = json.load(f)

    reactions: Dict[str, Dict[str, Any]] = {}
    for smarts, payload in raw_reactions.items():
        drop_bools = payload.get("drop_bools", [False, False])
        reactions[smarts] = {
            "drop_bools": [bool(drop_bools[0]), bool(drop_bools[1])],
            "index": int(payload["index"]),
        }

    n_reactions = len(reactions)

    compat_path = vocab_dir / "compatibility.pt"
    compatibility = torch.load(compat_path)
    assert compatibility.dim() == 3, "compatibility.pt must be 3D"
    compatibility = compatibility[:n_building_blocks, :n_reactions]
    n_centers = compatibility.shape[2]

    frag_path = vocab_dir / "fragment_features.pt"
    meta_path = vocab_dir / "meta.json"
    frag = torch.load(frag_path)
    fragment_maccs = frag["fragment_maccs"]
    fragment_atomfeats = frag["fragment_atomfeats"]
    fragment_bondfeats = frag["fragment_bondfeats"]
    fragment_atomadj = frag["fragment_atomadj"]

    meta = json.loads(meta_path.read_text())
    max_atoms_per_bb = int(meta["max_atoms_per_bb"])
    n_atom_features = int(meta["n_atom_features"])
    n_bond_features = int(meta["n_bond_features"])
    atom_types = meta["atom_types"]

    _VOCAB = {
        "BUILDING_BLOCKS": raw_bbs,
        "BUILDING_BLOCKS_SMI_TO_IDX": bb_smi_to_idx,
        "BUILDING_BLOCKS_IDX_TO_SMI": bb_idx_to_smi,
        "N_BUILDING_BLOCKS": n_building_blocks,
        "MAX_ATOMS_PER_BB": max_atoms_per_bb,
        "FRAGMENT_MACCS": fragment_maccs,
        "FRAGMENT_ATOMFEATS": fragment_atomfeats,
        "FRAGMENT_BONDFEATS": fragment_bondfeats,
        "FRAGMENT_ATOMADJ": fragment_atomadj,
        "ATOM_TYPES": atom_types,
        "N_ATOM_FEATURES": n_atom_features,
        "N_BOND_FEATURES": n_bond_features,
        "REACTIONS": reactions,
        "N_REACTIONS": n_reactions,
        "N_CENTERS": n_centers,
        "COMPATIBILITY": compatibility,
    }

    BUILDING_BLOCKS = _VOCAB["BUILDING_BLOCKS"]
    BUILDING_BLOCKS_SMI_TO_IDX = _VOCAB["BUILDING_BLOCKS_SMI_TO_IDX"]
    BUILDING_BLOCKS_IDX_TO_SMI = _VOCAB["BUILDING_BLOCKS_IDX_TO_SMI"]
    N_BUILDING_BLOCKS = _VOCAB["N_BUILDING_BLOCKS"]
    MAX_ATOMS_PER_BB = _VOCAB["MAX_ATOMS_PER_BB"]
    FRAGMENT_MACCS = _VOCAB["FRAGMENT_MACCS"]
    FRAGMENT_ATOMFEATS = _VOCAB["FRAGMENT_ATOMFEATS"]
    FRAGMENT_BONDFEATS = _VOCAB["FRAGMENT_BONDFEATS"]
    FRAGMENT_ATOMADJ = _VOCAB["FRAGMENT_ATOMADJ"]
    ATOM_TYPES = _VOCAB["ATOM_TYPES"]
    N_ATOM_FEATURES = _VOCAB["N_ATOM_FEATURES"]
    N_BOND_FEATURES = _VOCAB["N_BOND_FEATURES"]
    REACTIONS = _VOCAB["REACTIONS"]
    N_REACTIONS = _VOCAB["N_REACTIONS"]
    N_CENTERS = _VOCAB["N_CENTERS"]
    COMPATIBILITY = _VOCAB["COMPATIBILITY"]

    return _VOCAB


def get_vocab() -> Dict[str, Any]:
    """Get loaded vocabulary or raise if not initialized."""
    if _VOCAB is None:
        raise RuntimeError(
            "Vocabulary not initialized. Call load_vocabulary(vocab_dir) first."
        )
    return _VOCAB


def is_vocab_loaded() -> bool:
    return _VOCAB is not None


COORDS_STD = 2.7962567753408374


SMILES_PATH = ROOT_DIR / "vocabulary/smiles_train_full.txt"
try:
    with open(SMILES_PATH) as f:
        TRAIN_SMILES = [s.strip() for s in f.readlines()]
except FileNotFoundError:
    warnings.warn(f"Training SMILES not found at {SMILES_PATH}.")
    TRAIN_SMILES = []

N_PHARM = 8
MAX_PHARM = 40
