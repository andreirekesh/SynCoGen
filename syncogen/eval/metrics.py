"""Evaluate generated molecules (2D SMILES or 3D SDF/pickle).

Computes validity, uniqueness, novelty, diversity, QED, SA score, single-point
energies (xTB/MMFF) and optional retrosynthesis success (AiZynthFinder /
Syntheseus) and PoseBusters validity. Run ``python -m syncogen.eval.metrics
--help`` for the CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

from syncogen.eval.energy import compute_energies
from syncogen.eval.synthesizability import Synthesizability

try:
    from posebusters import PoseBusters
except ImportError:
    PoseBusters = None

logger = logging.getLogger(__name__)


# ------------------------------ diversity --------------------------------- #
def _internal_diversity(fps: Sequence) -> float:
    """1 - mean pairwise Tanimoto similarity."""
    total, pairs = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(sims)
        pairs += len(sims)
    return 1.0 - total / pairs if pairs else 0.0


def _cluster_diversity(fps: Sequence, thresh: float = 0.65) -> float:
    """Fraction of molecules surviving Tanimoto sphere-exclusion clustering."""
    if not fps:
        return 0.0
    picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, len(fps), thresh)
    return len(picks) / len(fps)


# ------------------------------- helpers ---------------------------------- #
def _embed_conformer(mol: Chem.Mol, seed: int = 0xF00D) -> bool:
    """Generate one 3D conformer (ETKDG + MMFF) in place."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return True


def _per_atom(energy):
    """Divide (total, nci, nbi) by atom count, or return Nones if missing."""
    if energy and energy[3]:
        total, nci, nbi, n = energy
        return total / n, nci / n, nbi / n
    return None, None, None


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


# ------------------------------- evaluate --------------------------------- #
def evaluate(
    entries: Sequence[Union[str, Path, Chem.Mol]],
    *,
    input_mode: str = "smiles",
    xtb_versions: Sequence[str] = ("mmff", "gfnff", "2"),
    xtb_path: str = "xtb",
    diversity_thresh: float = 0.65,
    training_smiles: Optional[Sequence[str]] = None,
    num_workers: int = 8,
    posebuster: bool = False,
    aizynth: bool = False,
    aizynth_config: Optional[str] = None,
    syntheseus: bool = False,
    syntheseus_inventory: Optional[Sequence[str]] = None,
    syntheseus_model: str = "local_retro",
    subsample: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate generated molecules and return a dictionary of metrics."""
    # ---- collect raw entries ----
    if input_mode == "sdf":
        expanded: List[Any] = []
        total = 0
        for entry in entries:
            if isinstance(entry, Chem.Mol):
                total += 1
                expanded.append(entry)
            else:
                supplier = Chem.SDMolSupplier(str(entry), removeHs=False)
                total += len(supplier)
                expanded.extend(supplier)
        entries = expanded
    else:
        total = len(entries)

    if subsample is not None and subsample < len(entries):
        random.seed(seed)
        entries = random.sample(list(entries), subsample)
        total = len(entries)

    entries = [e for e in entries if e is not None]

    # ---- validate & prepare 3D coordinates ----
    need_3d = len(list(xtb_versions)) > 0
    mols: List[Chem.Mol] = []
    smiles: List[str] = []
    for entry in tqdm(entries, desc="Validating"):
        if input_mode == "smiles":
            mol = Chem.MolFromSmiles(str(entry))
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            if need_3d and mol.GetNumConformers() == 0:
                _embed_conformer(mol)  # energy is skipped later if this fails
        else:  # sdf / pkl: RDKit mol that should already carry 3D coords
            mol = Chem.Mol(entry)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            mol = Chem.AddHs(mol, addCoords=True)
        mols.append(mol)
        smiles.append(Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True))

    validity = 100.0 * len(mols) / total if total else 0.0
    logger.info("Validity: %.2f%% (%d/%d)", validity, len(mols), total)

    # ---- single-point energies ----
    energy: Dict[str, list] = {}
    energy_per_atom: Dict[str, list] = {}
    avg_energy_per_atom: Dict[str, list] = {}
    if need_3d and mols:
        scratch = tempfile.mkdtemp(prefix="syncogen_eval_")
        try:
            for version in xtb_versions:
                energy[version] = compute_energies(
                    mols, xtb_version=version, xtb_path=xtb_path,
                    out_dir=str(Path(scratch) / version), n_jobs=num_workers,
                )
                energy_per_atom[version] = [_per_atom(e) for e in energy[version]]
                avg_energy_per_atom[version] = [
                    float(np.nanmean([row[i] for row in energy_per_atom[version] if row[i] is not None]))
                    for i in range(3)
                ]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        logger.info("Mean energy per atom: %s", avg_energy_per_atom)

    # ---- diversity & uniqueness ----
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(m) for m in mols]
    intdiv = _internal_diversity(fps)
    cluster_div = _cluster_diversity(fps, diversity_thresh)
    uniqueness = 100.0 * len(set(smiles)) / len(smiles) if smiles else 0.0

    # ---- novelty ----
    exact_novelty = num_novel = None
    if training_smiles:
        train = set()
        for smi in tqdm(training_smiles, desc="Canonicalising training set"):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                train.add(Chem.MolToSmiles(m, canonical=True))
        novel = [s for s in smiles if s not in train]
        num_novel = len(novel)
        exact_novelty = 100.0 * num_novel / len(smiles) if smiles else 0.0
        logger.info("Exact novelty: %.2f%%", exact_novelty)

    # ---- PoseBusters validity (3D geometry sanity; over all input molecules) ----
    pb_valid = None
    if posebuster:
        if PoseBusters is None:
            logger.warning("posebusters not installed; skipping PoseBusters validity.")
        else:
            report = PoseBusters(config="mol", max_workers=num_workers).bust(
                [Chem.RemoveHs(m) for m in mols], None, None
            )
            pb_valid = float(report.all(axis=1).sum() / total) if total else 0.0
            logger.info("PoseBusters validity: %s", pb_valid)

    # ---- QED ----
    qed = [QED.qed(m) for m in tqdm(mols, desc="QED")]

    # ---- synthesizability ----
    synth = Synthesizability(
        aizynth=aizynth, aizynth_config=aizynth_config,
        syntheseus=syntheseus, syntheseus_inventory=syntheseus_inventory,
        syntheseus_model=syntheseus_model,
    )
    synth_results = synth(mols)
    aizynth_solved = [r["aizynth_solved"] for r in synth_results if r["aizynth_solved"] is not None]
    syntheseus_solved = [r["syntheseus_solved"] for r in synth_results if r["syntheseus_solved"] is not None]

    return {
        "validity_%": validity,
        "uniqueness_%": uniqueness,
        "exact_novelty_%": exact_novelty,
        "intdiv": intdiv,
        "cluster_diversity_%": 100.0 * cluster_div,
        "avg_qed": _mean(qed),
        "avg_sa_score": _mean(r["sa_score"] for r in synth_results),
        "avg_aizynth_score": _mean(r["aizynth_score"] for r in synth_results),
        "aizynth_valid": (sum(1 for v in aizynth_solved if v) / total)
        if (aizynth_solved and total) else None,
        "syntheseus_valid": (sum(1 for v in syntheseus_solved if v) / total)
        if (syntheseus_solved and total) else None,
        "pb_valid": pb_valid,
        "avg_energy_per_atom": avg_energy_per_atom,
        "energy_per_atom": energy_per_atom,
        "energy": energy,
        "num_input": total,
        "num_valid": len(mols),
        "num_unique": len(set(smiles)),
        "num_novel": num_novel,
    }


# --------------------------------- CLI ------------------------------------ #
def _read_entries(path: str, mode: str):
    if mode == "smiles":
        with open(path, encoding="utf-8") as f:
            if path.endswith(".csv"):
                return [row["SMILES"] for row in csv.DictReader(f)]
            return [line.split()[0] for line in f if line.strip()]
    if mode == "sdf":
        p = Path(path)
        return [p] if p.is_file() else sorted(p.glob("*.sdf"))
    if mode == "pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unknown input mode: {mode}")


def _read_smiles_column(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "smiles" if "smiles" in (reader.fieldnames or []) else "SMILES"
        return [row[col] for row in reader if row.get(col)]


def _read_inventory(paths: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not paths:
        return None
    smiles: List[str] = []
    for path in paths:
        smiles.extend(_read_smiles_column(path) or [])
    logger.info("Loaded %d building blocks from %d file(s)", len(smiles), len(paths))
    return smiles


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Evaluate generated molecules.")
    p.add_argument("--input", required=True, help="SMILES CSV/file, SDF file/dir, or pickle of mols")
    p.add_argument("--input_mode", choices=["smiles", "sdf", "pkl"], default="smiles")
    p.add_argument("--out_json", default="results.json")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--subsample", type=int, default=None, help="Randomly evaluate at most N molecules")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xtb_versions", nargs="*", default=["mmff", "gfnff", "2"],
                   help="Energy back-ends (mmff, gfnff, 0/1/2). Pass 'none' to skip energy.")
    p.add_argument("--xtb_path", default="xtb")
    p.add_argument("--diversity_thresh", type=float, default=0.65)
    p.add_argument("--training_csv", help="Training-set CSV with a SMILES column (for novelty)")
    p.add_argument("--posebuster", action="store_true", help="Compute PoseBusters validity (3D inputs)")
    p.add_argument("--aizynth", action="store_true")
    p.add_argument("--aizynth_config", help="AiZynthFinder YAML config")
    p.add_argument("--syntheseus", action="store_true")
    p.add_argument("--syntheseus_inventory", nargs="*", default=None,
                   help="CSV file(s) with a 'smiles' column of purchasable building blocks")
    p.add_argument("--syntheseus_model", choices=["local_retro", "root_aligned"], default="local_retro")
    args = p.parse_args()

    results = evaluate(
        _read_entries(args.input, args.input_mode),
        input_mode=args.input_mode,
        xtb_versions=[v for v in args.xtb_versions if v.lower() != "none"],
        xtb_path=args.xtb_path,
        diversity_thresh=args.diversity_thresh,
        training_smiles=_read_smiles_column(args.training_csv),
        num_workers=args.num_workers,
        posebuster=args.posebuster,
        aizynth=args.aizynth,
        aizynth_config=args.aizynth_config,
        syntheseus=args.syntheseus,
        syntheseus_inventory=_read_inventory(args.syntheseus_inventory),
        syntheseus_model=args.syntheseus_model,
        subsample=args.subsample,
        seed=args.seed,
    )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=_json_default)
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
