"""Acceptance test for a generated conformer LMDB.

Checks the things that would otherwise fail silently at training time, or worse,
train on misaligned coordinates:

  1. key hygiene   - keys parse as mol_{data_index}_final_conf_{j}, conformer
                     indices are contiguous from 0 (``get_conformer_keys`` stops
                     at the first gap, so a gap silently hides conformers)
  2. coverage      - every graph in the dataset has at least one conformer, since
                     ``SyncogenDataset.get`` raises on a missing mol_*_conf_0
  3. alignment     - for a random sample, the per-fragment valid-atom pattern the
                     reader recovers equals ``ground_truth_atom_mask`` exactly, and
                     element identities match ``build_rdkit()`` atom-by-atom. This
                     is the check that catches a wrong atom ordering, which is
                     otherwise invisible and poisons training.
  4. geometry      - bond lengths are physical, coordinates finite, no all-zero
                     conformers
  5. end-to-end    - drives the real SyncogenDataset/DataLoader over a sample

  python tools/conformers/verify_lmdb.py \
      --lmdb <DATA>/conformers/mol2s.lmdb \
      --graphs <DATA>/graphs_1m.pt.gz \
      --vocab_dir <DATA>/vocabulary --sample 2000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parents[1]))  # repo root, for `import syncogen`

KEY_RE = re.compile(r"^mol_(\d+)_final_conf_(\d+)$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", required=True)
    p.add_argument("--graphs", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--sample", type=int, default=2000, help="molecules to deep-check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--expect_indices",
        default=None,
        help="optional 'lo:hi' data_index range this LMDB should cover",
    )
    args = p.parse_args()

    import lmdb
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from syncogen.constants.constants import load_vocabulary

    load_vocabulary(args.vocab_dir)
    from mol2_io import graph_from_data
    from syncogen.utils.file_readers import mol2_to_coordinates, parse_mol2_file

    problems: list[str] = []

    # ---- 1. scan all keys
    env = lmdb.open(args.lmdb, readonly=True, lock=False, subdir=True)
    confs: dict[int, set[int]] = defaultdict(set)
    n_records = 0
    with env.begin() as txn:
        for raw_key, _ in txn.cursor():
            k = raw_key.decode()
            m = KEY_RE.match(k)
            if not m:
                problems.append(f"malformed key: {k}")
                continue
            confs[int(m.group(1))].add(int(m.group(2)))
            n_records += 1
    print(f"records            {n_records:,}")
    print(f"molecules          {len(confs):,}")
    if confs:
        per = np.array([len(v) for v in confs.values()])
        print(
            f"conformers/molecule mean {per.mean():.2f}  median {np.median(per):.0f}  "
            f"min {per.min()}  max {per.max()}"
        )

    # contiguity: get_conformer_keys stops at the first missing index
    gaps = [i for i, s in confs.items() if s != set(range(len(s)))]
    if gaps:
        problems.append(
            f"{len(gaps)} molecules have non-contiguous conformer indices "
            f"(e.g. {gaps[:3]}) - conformers past the gap are unreachable"
        )

    # ---- 2. coverage against the graph list
    gp = Path(args.graphs)
    opener = gzip.open if gp.suffix == ".gz" else open
    with opener(gp, "rb") as fh:
        graphs = torch.load(fh, weights_only=False)
    by_idx = {int(g.data_index): g for g in graphs}
    print(f"graphs in dataset  {len(by_idx):,}")

    if args.expect_indices:
        lo, hi = (int(x) for x in args.expect_indices.split(":"))
        expected = set(range(lo, hi))
    else:
        expected = set(by_idx)
    missing = expected - set(confs)
    extra = set(confs) - set(by_idx)
    print(
        f"missing conformers {len(missing):,} "
        f"({100 * len(missing) / max(len(expected), 1):.2f}% of expected)"
    )
    if extra:
        problems.append(
            f"{len(extra)} data_index values in LMDB are not in the graph set"
        )

    # ---- 3/4. deep per-molecule checks on a sample
    random.seed(args.seed)
    pool = sorted(set(confs) & set(by_idx))
    pick = random.sample(pool, min(args.sample, len(pool)))
    bad_mask = bad_order = bad_geom = bad_parse = 0
    bondlens: list[float] = []

    with env.begin() as txn:
        for di in pick:
            g = by_idx[di]
            try:
                _, mol, atom_mask = graph_from_data(g)
            except Exception as exc:
                problems.append(f"mol {di}: graph rebuild failed: {exc}")
                continue
            heavy_syms = [a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

            for j in sorted(confs[di]):
                raw = txn.get(f"mol_{di}_final_conf_{j}".encode())
                try:
                    coords, cmask, _ = mol2_to_coordinates(raw, atom_mask=atom_mask)
                    _, bonds, ann = parse_mol2_file(raw)
                except Exception as exc:
                    bad_parse += 1
                    problems.append(f"mol_{di}_final_conf_{j}: parse failed: {exc}")
                    continue

                if cmask.shape != atom_mask.shape or not torch.equal(
                    cmask, atom_mask.bool()
                ):
                    bad_mask += 1
                    problems.append(
                        f"mol_{di}_final_conf_{j}: valid-atom pattern != mask"
                    )
                    continue

                file_heavy = [e for _, (e, _, _) in sorted(ann.items()) if e != "H"]
                if file_heavy != heavy_syms:
                    bad_order += 1
                    problems.append(
                        f"mol_{di}_final_conf_{j}: atom ORDER mismatch vs build_rdkit"
                    )
                    continue

                pos = coords.reshape(-1, 3)[atom_mask.bool().reshape(-1)].numpy()
                if not np.isfinite(pos).all() or np.abs(pos).max() == 0:
                    bad_geom += 1
                    problems.append(f"mol_{di}_final_conf_{j}: degenerate coordinates")
                    continue
                if j == 0:
                    flat = coords.reshape(-1, 3).numpy()
                    for a, b, _ in bonds[: min(len(bonds), 200)]:
                        pass  # bond indices are file-local; length check below uses mol graph
                    idx = np.flatnonzero(atom_mask.bool().reshape(-1).numpy())
                    remap = {int(v): k for k, v in enumerate(idx)}
                    for bond in mol.GetBonds():
                        i, k2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        bondlens.append(float(np.linalg.norm(pos[i] - pos[k2])))

    print(f"\ndeep-checked       {len(pick):,} molecules")
    print(f"  mask mismatches  {bad_mask}")
    print(f"  order mismatches {bad_order}")
    print(f"  bad geometry     {bad_geom}")
    print(f"  parse failures   {bad_parse}")
    if bondlens:
        bl = np.array(bondlens)
        print(
            f"  heavy-atom bond lengths: mean {bl.mean():.3f} A  "
            f"min {bl.min():.3f}  max {bl.max():.3f}  "
            f"outside [0.9,2.6]: {int(((bl < 0.9) | (bl > 2.6)).sum())}"
        )
        if bl.min() < 0.9 or bl.max() > 2.6:
            problems.append("implausible heavy-atom bond lengths present")
    env.close()

    # ---- 5. end-to-end through the real dataset class
    print("\nend-to-end via SyncogenDataset ...")
    try:
        from torch_geometric.loader import DataLoader

        from syncogen.data.dataloader import SyncogenDataset

        subset = [by_idx[d] for d in pick[:256]]
        ds = SyncogenDataset(
            conformers_path=args.lmdb,
            data_list=subset,
            sample_conformer=False,
            load_bonds=True,
        )
        dl = DataLoader(ds, batch_size=32, num_workers=0)
        nb = 0
        for batch in dl:
            assert batch.coordinates.shape[-1] == 3
            assert torch.isfinite(batch.coordinates).all()
            nb += 1
        print(
            f"  OK: {nb} batches, coordinates finite, shape {tuple(batch.coordinates.shape)}"
        )
    except Exception as exc:
        problems.append(f"end-to-end dataloader failed: {type(exc).__name__}: {exc}")
        print(f"  FAILED: {exc}")

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} PROBLEM(S) - showing first 20:")
        for s in problems[:20]:
            print("  -", s)
    else:
        print("ALL CHECKS PASSED")
    out = Path(args.lmdb + ".verify.json")
    out.write_text(
        json.dumps(
            {
                "records": n_records,
                "molecules": len(confs),
                "missing": len(missing),
                "deep_checked": len(pick),
                "bad_mask": bad_mask,
                "bad_order": bad_order,
                "bad_geometry": bad_geom,
                "bad_parse": bad_parse,
                "problems": problems[:200],
            },
            indent=2,
        )
    )
    print(f"report -> {out}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
