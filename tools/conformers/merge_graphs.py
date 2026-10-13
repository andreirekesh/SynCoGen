"""Merge per-length graph files into the single dataset the loader reads.

``generate_graph_data.py`` writes one file per fragment count, and each run
numbers ``data_index`` from 0 and dedups only against its own ``seen_smiles``.
``SyncogenDataManager`` loads ONE file and splits it POSITIONALLY. So three
things have to happen before training, none of which generation does:

  1. cross-file dedup. Uniqueness is only guaranteed within a run, so two runs
     can emit the same molecule. For 2-fragment molecules the reachable space is
     small enough for this to matter (~0.06-2.5% of a 1M stratum); for longer
     chains it is negligible, but the merge is exact either way.
  2. renumber ``data_index`` to 0..N-1 over the merged list. Conformer keys are
     ``mol_{data_index}_final_conf_{j}``, so overlapping indices from different
     runs would collide in a single LMDB. Generate conformers AFTER merging, or
     the keys will not match.
  3. interleave the strata. Concatenating leaves the file sorted by fragment
     count, and a positional split then puts only the longest molecules in
     validation and removes them from training entirely.

  python tools/conformers/merge_graphs.py \
      --inputs <SCRATCH>/graphs/len2_1M.pt <SCRATCH>/graphs/len3_1M.pt \
               <SCRATCH>/graphs/len4_1M.pt <SCRATCH>/graphs/len5_1M.pt \
      --out <DATA>/graphs_4m.pt --per_bucket_limit 1000000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        return torch.load(fh, weights_only=False)


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as fh:
        torch.save(obj, fh)


def n_fragments(g) -> int:
    """One node per building block, so the node count IS the fragment count."""
    return int(g.x.shape[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--per_bucket_limit",
        type=int,
        default=0,
        help="keep at most this many molecules per fragment-count bucket "
        "(after dedup); 0 keeps everything",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no_interleave",
        action="store_true",
        help="keep source order; only do this if you shuffle later",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    buckets: dict[int, list] = defaultdict(list)
    seen: set[str] = set()
    per_file: list[dict] = []

    for raw in args.inputs:
        path = Path(raw)
        graphs = load(path)
        kept = dup = 0
        counts: Counter = Counter()
        for g in graphs:
            smi = getattr(g, "smiles", None)
            if smi is None:
                raise RuntimeError(
                    f"{path}: record has no .smiles, cannot dedup exactly"
                )
            if smi in seen:
                dup += 1
                continue
            seen.add(smi)
            nf = n_fragments(g)
            buckets[nf].append(g)
            counts[nf] += 1
            kept += 1
        per_file.append(
            {
                "file": str(path),
                "records": len(graphs),
                "kept": kept,
                "duplicates_dropped": dup,
                "by_fragment_count": dict(sorted(counts.items())),
            }
        )
        print(
            f"{path.name:>24}  {len(graphs):>9,} in  {kept:>9,} kept  {dup:>7,} dup  "
            f"frags={dict(sorted(counts.items()))}",
            flush=True,
        )

    # truncate each bucket to the requested size, sampling rather than head-cutting
    # so the kept subset is not biased by generation order
    if args.per_bucket_limit:
        for nf, lst in buckets.items():
            if len(lst) > args.per_bucket_limit:
                rng.shuffle(lst)
                del lst[args.per_bucket_limit :]

    # Interleave so that ANY contiguous slice holds every fragment count in
    # roughly its global proportion. Round-robin is not enough: it exhausts the
    # smaller buckets first and leaves a tail of pure largest-bucket, which is
    # the same validation-split problem in a subtler form. Instead give each
    # record a position (rank + 0.5) / bucket_size, which spreads every bucket
    # uniformly over [0, 1) regardless of how unequal the buckets are.
    merged: list = []
    if args.no_interleave:
        for nf in sorted(buckets):
            merged.extend(buckets[nf])
    else:
        placed = []
        for nf in sorted(buckets):
            lst = buckets[nf]
            rng.shuffle(lst)
            n = len(lst)
            for rank, g in enumerate(lst):
                placed.append(((rank + 0.5) / n, nf, g))
        placed.sort(key=lambda t: (t[0], t[1]))
        merged = [g for _, _, g in placed]

    for i, g in enumerate(merged):
        g.data_index = i

    out = Path(args.out)
    save(merged, out)

    summary = {
        "output": str(out),
        "molecules": len(merged),
        "unique_smiles": len(seen),
        "duplicates_dropped": sum(f["duplicates_dropped"] for f in per_file),
        "by_fragment_count": {k: len(v) for k, v in sorted(buckets.items())},
        "per_bucket_limit": args.per_bucket_limit,
        "interleaved": not args.no_interleave,
        "inputs": per_file,
    }
    Path(str(out) + ".merge.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(merged):,} molecules -> {out}")
    print(f"  dropped {summary['duplicates_dropped']:,} cross-file duplicates")
    print(f"  per fragment count: {summary['by_fragment_count']}")
    print(f"  data_index renumbered 0..{len(merged) - 1}")
    print(f"  report -> {out}.merge.json")


if __name__ == "__main__":
    main()
