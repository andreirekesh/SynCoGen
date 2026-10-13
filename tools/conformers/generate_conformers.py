"""Conformer generation for SynCoGen training data, reproducing the SynCoGen paper.

Pipeline (arXiv 2507.11818 sec. 3.2, verbatim):
  "For each molecular graph, 50 initial conformers were generated using the ETKDG
   algorithm (RDKit implementation). These structures were energy-minimized using
   the MMFF94 force field, and all conformers within 10 kcal/mol of the global
   minimum were retained. The resulting geometries were then re-optimized with the
   semi-empirical GFN2-xTB method, after which the same 10 kcal/mol energy
   threshold was applied. At every stage, redundant structures were removed by
   geometry-based clustering (RMSD<1.5A)."
  -> "yields, on average, 5.4 distinct conformers per graph."

Output is one LMDB shard whose keys are ``mol_{data_index}_final_conf_{j}`` and
whose values are MOL2 text, conformers ordered by ascending final energy so that
``_conf_0`` (what the loader takes when ``sample_conformer=False``) is the lowest
energy geometry. See mol2_io.py for the atom/fragment indexing contract.

Run one shard:
  python tools/conformers/generate_conformers.py \
      --graphs <DATA>/graphs_1m.pt.gz \
      --vocab_dir <DATA>/vocabulary \
      --out <SCRATCH>/conformers/shards/shard_000.lmdb \
      --shard 0 --num_shards 256 --workers 64
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Running `python tools/conformers/generate_conformers.py` puts the SCRIPT's
# directory on sys.path, not the repo root, so `import syncogen` would fail in
# every pool worker (and Pool silently respawns workers whose initializer raises,
# spinning forever). Put the repo root on the path explicitly.
sys.path.insert(1, str(_HERE.parents[1]))

HARTREE_KCAL = 627.5094740631
# Covalent radii (A) for the connectivity guard; covers this vocabulary's elements.
COVALENT_R = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

_VOCAB_LOADED = False


def init_worker(vocab_dir: str) -> None:
    """Per-process setup: silence RDKit, pin BLAS/OMP to 1 thread, load vocabulary.

    The vocabulary populates module-level constants (MAX_ATOMS_PER_BB, building
    block tables), so it must be loaded in every worker process before any
    syncogen import that reads those constants.
    """
    global _VOCAB_LOADED
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not _VOCAB_LOADED:
        from syncogen.constants.constants import load_vocabulary

        load_vocabulary(vocab_dir)
        _VOCAB_LOADED = True


# ---------------------------------------------------------------- xTB


def xtb_optimize(
    symbols: Sequence[str],
    coords: np.ndarray,
    charge: int,
    xtb_bin: str,
    scratch_root: Optional[str],
    accuracy: Optional[float] = None,
    opt_level: str = "normal",
    timeout: int = 600,
) -> Optional[Tuple[np.ndarray, float]]:
    """GFN2-xTB geometry optimization. Returns (coords, energy in Hartree) or None.

    Atom order is preserved: xtb's xtbopt.xyz lists atoms in input order, which is
    what keeps the MOL2 fragment/atom correspondence intact.
    """
    workdir = tempfile.mkdtemp(prefix="xtb_", dir=scratch_root)
    try:
        xyz = Path(workdir) / "in.xyz"
        with open(xyz, "w") as fh:
            fh.write(f"{len(symbols)}\n\n")
            for s, (x, y, z) in zip(symbols, coords):
                fh.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")

        cmd = [
            xtb_bin,
            "in.xyz",
            "--opt",
            opt_level,
            "--gfn",
            "2",
            "--chrg",
            str(charge),
            "--uhf",
            "0",
            "--parallel",
            "1",
        ]
        if accuracy is not None:
            cmd += ["--acc", str(accuracy)]

        env = dict(os.environ)
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_STACKSIZE="1G",
            XTBPATH=workdir,
            XTBHOME=workdir,
        )
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out = Path(workdir) / "xtbopt.xyz"
        if proc.returncode != 0 or not out.exists():
            return None

        lines = out.read_text().splitlines()
        n = int(lines[0].split()[0])
        # comment line: "energy: -123.45678901 gnorm: 0.0001 xtb: 6.7.1"
        energy = None
        toks = lines[1].split()
        for i, t in enumerate(toks):
            if t.startswith("energy"):
                energy = float(toks[i + 1])
                break
        if energy is None:
            return None
        new = np.array(
            [[float(v) for v in ln.split()[1:4]] for ln in lines[2 : 2 + n]],
            dtype=float,
        )
        if new.shape != coords.shape:
            return None
        return new, energy
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def connectivity_preserved(
    mol, coords: np.ndarray, slack_bond: float = 1.35, slack_nonbond: float = 0.75
) -> bool:
    """Reject geometries where xTB broke or formed a bond.

    The MOL2 bond block is taken from the input graph, so a geometry whose
    connectivity no longer matches it would be mislabelled training data. Checks
    bonded heavy pairs are not stretched past ``slack_bond`` x (r_i + r_j) and
    non-bonded heavy pairs are not compressed below ``slack_nonbond`` x (r_i + r_j).
    """
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(idx) < 2:
        return True
    radii = np.array(
        [COVALENT_R.get(mol.GetAtomWithIdx(i).GetSymbol(), 0.8) for i in idx]
    )
    pos = coords[idx]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    rsum = radii[:, None] + radii[None, :]

    local = {i: k for k, i in enumerate(idx)}
    bonded = np.zeros_like(d, dtype=bool)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in local and j in local:
            bonded[local[i], local[j]] = bonded[local[j], local[i]] = True

    if (d[bonded] > slack_bond * rsum[bonded]).any():
        return False
    off = ~bonded
    np.fill_diagonal(off, False)
    return not (d[off] < slack_nonbond * rsum[off]).any()


# ---------------------------------------------------------------- pruning


def prune_by_rmsd(mol, conf_ids: List[int], threshold: float) -> List[int]:
    """Greedy geometry clustering: keep a conformer only if it is >= threshold A
    from every conformer already kept. ``conf_ids`` must be pre-sorted by energy so
    the representative of each cluster is its lowest-energy member.

    RMSD is heavy-atom only (hydrogens are not part of the learned coordinates).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    heavy = Chem.RemoveHs(Chem.Mol(mol))
    kept: List[int] = []
    for cid in conf_ids:
        if all(
            AllChem.GetConformerRMS(heavy, cid, k, prealigned=False) >= threshold
            for k in kept
        ):
            kept.append(cid)
    return kept


def energy_window(
    conf_ids: List[int], energies: Dict[int, float], window: float
) -> List[int]:
    """Keep conformers within ``window`` kcal/mol of the global minimum, sorted
    by ascending energy."""
    if not conf_ids:
        return []
    lo = min(energies[c] for c in conf_ids)
    keep = [c for c in conf_ids if energies[c] - lo <= window]
    return sorted(keep, key=lambda c: energies[c])


# ---------------------------------------------------------------- per molecule


def process_molecule(args) -> Tuple[int, List[Tuple[str, bytes]], Dict]:
    """Full pipeline for one graph. Returns (data_index, [(key, mol2 bytes)], stats)."""
    (data, cfg) = args
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from mol2_io import (
        annotate_hydrogens,
        graph_from_data,
        verify_roundtrip,
        write_mol2,
    )

    di = int(data.data_index)
    stats = {
        "data_index": di,
        "embedded": 0,
        "after_mmff": 0,
        "xtb_ok": 0,
        "final": 0,
        "status": "ok",
    }
    try:
        _, mol, atom_mask = graph_from_data(data)
        if mol is None:
            stats["status"] = "build_failed"
            return di, [], stats
        if mol.GetNumAtoms() != int(atom_mask.sum()):
            stats["status"] = "mask_mismatch"
            return di, [], stats

        molH = Chem.AddHs(mol)
        annotate_hydrogens(molH)

        # --- stage 1: ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = cfg["seed"] + di
        params.pruneRmsThresh = cfg["embed_prune"]
        params.numThreads = 1
        params.useSmallRingTorsions = True
        cids = list(AllChem.EmbedMultipleConfs(molH, cfg["num_conformers"], params))
        if not cids:
            # retry once with random coords for hard cages
            params.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(molH, cfg["num_conformers"], params))
        if not cids:
            stats["status"] = "embed_failed"
            return di, [], stats
        stats["embedded"] = len(cids)

        # --- stage 2: MMFF94, energy window, RMSD prune
        mmff = AllChem.MMFFOptimizeMoleculeConfs(
            molH, maxIters=cfg["mmff_iters"], mmffVariant="MMFF94", numThreads=1
        )
        e_mmff = {
            cid: res[1] for cid, res in zip(cids, mmff) if res[1] == res[1]
        }  # drop NaN
        cids = energy_window(list(e_mmff), e_mmff, cfg["energy_window"])
        cids = prune_by_rmsd(molH, cids, cfg["rmsd_threshold"])
        if cfg["max_xtb_confs"] > 0:
            cids = cids[: cfg["max_xtb_confs"]]
        stats["after_mmff"] = len(cids)
        if not cids:
            stats["status"] = "mmff_failed"
            return di, [], stats

        # --- stage 3: GFN2-xTB re-optimization
        symbols = [a.GetSymbol() for a in molH.GetAtoms()]
        charge = sum(a.GetFormalCharge() for a in molH.GetAtoms())
        e_xtb: Dict[int, float] = {}
        for cid in cids:
            conf = molH.GetConformer(cid)
            start = np.array(conf.GetPositions())
            res = xtb_optimize(
                symbols,
                start,
                charge,
                cfg["xtb_bin"],
                cfg["scratch_root"],
                accuracy=cfg["xtb_accuracy"],
                opt_level=cfg["xtb_opt_level"],
                timeout=cfg["xtb_timeout"],
            )
            if res is None:
                continue
            new, energy = res
            if cfg["check_connectivity"] and not connectivity_preserved(molH, new):
                continue
            for i in range(molH.GetNumAtoms()):
                conf.SetAtomPosition(i, new[i].tolist())
            e_xtb[cid] = energy * HARTREE_KCAL
        stats["xtb_ok"] = len(e_xtb)
        if not e_xtb:
            stats["status"] = "xtb_failed"
            return di, [], stats

        # --- stage 4: same 10 kcal/mol window, final RMSD prune
        final = energy_window(list(e_xtb), e_xtb, cfg["energy_window"])
        final = prune_by_rmsd(molH, final, cfg["rmsd_threshold"])
        stats["final"] = len(final)

        # --- write (conf_0 = lowest energy, which is what training reads by default)
        records: List[Tuple[str, bytes]] = []
        for j, cid in enumerate(final):
            key = f"mol_{di}_final_conf_{j}"
            text = write_mol2(molH, key, conf_id=cid)
            if cfg["verify"]:
                verify_roundtrip(text, molH, atom_mask, conf_id=cid)
            records.append((key, text.encode()))
        return di, records, stats

    except Exception as exc:  # one bad molecule must not kill the shard
        stats["status"] = f"error:{type(exc).__name__}:{exc}"[:200]
        return di, [], stats


# ---------------------------------------------------------------- shard driver


def load_shard(
    graphs_path: Path, shard: int, num_shards: int, strided: bool = True
) -> List:
    """Load the graph list and return this shard's slice.

    Strided (``data[shard::num_shards]``) by default, because aablock_1m_v2 is
    stored strictly sorted by fragment count (2-fragment molecules occupy indices
    0-299,999, then 3, then 4, then 5 from 900,000). Contiguous shards would
    therefore be badly load-imbalanced -- the 5-fragment shards cost several times
    a 2-fragment shard. Striding gives every task the same size mix.

    Keys are ``mol_{data_index}_...``, so the slicing scheme has no effect on the
    output keyspace and shards stay disjoint either way.
    """
    opener = gzip.open if graphs_path.suffix == ".gz" else open
    with opener(graphs_path, "rb") as fh:
        data = torch.load(fh, weights_only=False)
    if strided:
        return data[shard::num_shards]
    total = len(data)
    per = (total + num_shards - 1) // num_shards
    lo, hi = shard * per, min((shard + 1) * per, total)
    return data[lo:hi]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--graphs", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--out", required=True, help="output LMDB path for this shard")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="debug: only N molecules")
    p.add_argument(
        "--contiguous_shards",
        action="store_true",
        help="use contiguous instead of strided slices (see load_shard)",
    )
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))

    p.add_argument("--num_conformers", type=int, default=50, help="paper: 50")
    p.add_argument(
        "--energy_window", type=float, default=10.0, help="kcal/mol, paper: 10"
    )
    p.add_argument("--rmsd_threshold", type=float, default=1.5, help="A, paper: 1.5")
    p.add_argument(
        "--embed_prune",
        type=float,
        default=0.5,
        help="cheap RMSD prune inside ETKDG; <=0 disables",
    )
    p.add_argument("--mmff_iters", type=int, default=2000)
    p.add_argument(
        "--max_xtb_confs",
        type=int,
        default=0,
        help="cap conformers entering xTB (0 = no cap, faithful to paper)",
    )
    p.add_argument("--xtb_bin", default=os.environ.get("XTB_BIN", "xtb"))
    p.add_argument("--xtb_opt_level", default="normal")
    p.add_argument("--xtb_accuracy", type=float, default=None)
    p.add_argument("--xtb_timeout", type=int, default=600)
    p.add_argument("--no_check_connectivity", action="store_true")
    p.add_argument(
        "--no_verify",
        action="store_true",
        help="skip the per-record round-trip assertion (not recommended)",
    )
    p.add_argument("--map_size_gb", type=float, default=400.0)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument(
        "--scratch_root",
        default=os.environ.get("SLURM_TMPDIR") or None,
        help="node-local dir for xTB scratch; defaults to $SLURM_TMPDIR",
    )
    args = p.parse_args()

    import lmdb
    from multiprocessing import Pool

    # Fail fast in the parent: multiprocessing.Pool respawns any worker whose
    # initializer raises, so an import error inside a worker turns into a silent
    # infinite respawn loop instead of a crash.
    init_worker(args.vocab_dir)
    from mol2_io import graph_from_data  # noqa: F401

    t0 = time.time()
    data = load_shard(
        Path(args.graphs),
        args.shard,
        args.num_shards,
        strided=not args.contiguous_shards,
    )
    if args.limit:
        data = data[: args.limit]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(data)} molecules "
        f"loaded in {time.time() - t0:.0f}s",
        flush=True,
    )

    cfg = {
        "num_conformers": args.num_conformers,
        "energy_window": args.energy_window,
        "rmsd_threshold": args.rmsd_threshold,
        "embed_prune": args.embed_prune,
        "mmff_iters": args.mmff_iters,
        "max_xtb_confs": args.max_xtb_confs,
        "xtb_bin": args.xtb_bin,
        "xtb_opt_level": args.xtb_opt_level,
        "xtb_accuracy": args.xtb_accuracy,
        "xtb_timeout": args.xtb_timeout,
        "check_connectivity": not args.no_check_connectivity,
        "verify": not args.no_verify,
        "seed": args.seed,
        "scratch_root": args.scratch_root,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=int(args.map_size_gb * 1024**3), subdir=True)

    all_stats: List[Dict] = []
    n_conf = n_mol = 0
    t0 = time.time()
    with Pool(
        args.workers, initializer=init_worker, initargs=(args.vocab_dir,)
    ) as pool:
        pending: List[Tuple[str, bytes]] = []
        for k, (di, records, stats) in enumerate(
            pool.imap_unordered(process_molecule, ((d, cfg) for d in data), chunksize=1)
        ):
            all_stats.append(stats)
            pending.extend(records)
            n_conf += len(records)
            n_mol += bool(records)
            if len(pending) >= 2000:
                with env.begin(write=True) as txn:
                    for key, val in pending:
                        txn.put(key.encode(), val)
                pending.clear()
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(
                    f"  {k + 1}/{len(data)} mols  {n_conf} confs  "
                    f"{el / (k + 1):.2f} s/mol  eta {(len(data) - k - 1) * el / (k + 1) / 3600:.2f} h",
                    flush=True,
                )
        if pending:
            with env.begin(write=True) as txn:
                for key, val in pending:
                    txn.put(key.encode(), val)
    env.sync()
    env.close()

    stats_path = out.parent / f"{out.stem}_stats.json"
    from collections import Counter

    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "molecules_in": len(data),
        "molecules_with_conformers": n_mol,
        "conformers": n_conf,
        "conformers_per_molecule": n_conf / max(n_mol, 1),
        "wall_seconds": time.time() - t0,
        "status_counts": dict(Counter(s["status"].split(":")[0] for s in all_stats)),
        "mean_embedded": float(np.mean([s["embedded"] for s in all_stats])),
        "mean_after_mmff": float(np.mean([s["after_mmff"] for s in all_stats])),
        "mean_xtb_ok": float(np.mean([s["xtb_ok"] for s in all_stats])),
        "mean_final": float(np.mean([s["final"] for s in all_stats])),
        "config": {k: v for k, v in cfg.items() if k != "scratch_root"},
    }
    stats_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
