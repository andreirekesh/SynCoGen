"""Merge per-shard conformer LMDBs into the single LMDB the training loader reads.

``SyncogenDataManager.conformers_path`` points at one LMDB, so the shards produced
by the array job have to be concatenated. Keys are globally unique
(``mol_{data_index}_final_conf_{j}``) because shards partition data_index ranges,
so this is a straight copy with a duplicate check.

  python tools/conformers/merge_lmdb.py \
      --shards <SCRATCH>/conformers/shards \
      --out <DATA>/conformers/mol2s.lmdb

Writes 2 files (data.mdb, lock.mdb), which matters because /project is at its
inode quota.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

KEY_RE = re.compile(rb"^mol_(\d+)_final_conf_(\d+)$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="directory containing *.lmdb shards")
    p.add_argument("--out", required=True)
    p.add_argument("--map_size_gb", type=float, default=600.0)
    p.add_argument("--batch", type=int, default=20000)
    p.add_argument("--allow_duplicates", action="store_true")
    args = p.parse_args()

    import lmdb

    shards = sorted(Path(args.shards).glob("*.lmdb"))
    if not shards:
        raise SystemExit(f"no *.lmdb found under {args.shards}")
    print(f"merging {len(shards)} shards -> {args.out}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = lmdb.open(str(out), map_size=int(args.map_size_gb * 1024**3), subdir=True)

    total = dupes = 0
    per_mol: Counter = Counter()
    t0 = time.time()
    for s in shards:
        src = lmdb.open(str(s), readonly=True, lock=False, subdir=True)
        n = 0
        with src.begin() as rtxn:
            cur = rtxn.cursor()
            buf = []
            for key, val in cur:
                buf.append((key, val))
                m = KEY_RE.match(key)
                if m:
                    per_mol[int(m.group(1))] += 1
                if len(buf) >= args.batch:
                    with dst.begin(write=True) as wtxn:
                        for k, v in buf:
                            if not wtxn.put(k, v, overwrite=args.allow_duplicates):
                                dupes += 1
                    n += len(buf)
                    buf.clear()
            if buf:
                with dst.begin(write=True) as wtxn:
                    for k, v in buf:
                        if not wtxn.put(k, v, overwrite=args.allow_duplicates):
                            dupes += 1
                n += len(buf)
        src.close()
        total += n
        print(f"  {s.name}: {n:>9,} records  (running total {total:,})", flush=True)

    dst.sync()
    dst.close()

    counts = Counter(per_mol.values())
    summary = {
        "shards": len(shards),
        "records": total,
        "molecules": len(per_mol),
        "duplicate_keys_skipped": dupes,
        "mean_conformers_per_molecule": (
            (sum(per_mol.values()) / len(per_mol)) if per_mol else 0
        ),
        "conformers_per_molecule_histogram": dict(sorted(counts.items())),
        "wall_seconds": time.time() - t0,
    }
    print(json.dumps(summary, indent=2))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    if dupes and not args.allow_duplicates:
        print(f"WARNING: {dupes} duplicate keys were skipped - shards may overlap")


if __name__ == "__main__":
    main()
