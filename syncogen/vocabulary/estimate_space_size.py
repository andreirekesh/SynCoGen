# Estimate the size of the search space for a given vocabulary by simply raising
# the number of building blocks to the power of the average branching factor.

import json
import math
from pathlib import Path

import torch


def compute_avg_branching_factor(compat: torch.Tensor) -> float:
    """
    compat: tensor of shape [n_bbs, n_rxns, n_centers] with values {0,1,2,3}
    Returns average branching factor over all (rxn, center) pairs, where
    BF_{r,c} = (#role1-compatible BBs) * (#role2-compatible BBs).
    """
    assert compat.dim() == 3
    n_bbs, n_rxns, n_centers = compat.shape

    role1_mask = (compat == 1) | (compat == 3)
    role2_mask = (compat == 2) | (compat == 3)

    # [n_rxns, n_centers]
    n1 = role1_mask.sum(dim=0)
    n2 = role2_mask.sum(dim=0)

    # branching factor per (rxn, center)
    bf = (n1 * n2).float()  # shape [n_rxns, n_centers]

    avg_bf = bf.mean().item()
    return avg_bf


def estimate_k_bb_states(n_bbs: int, avg_bf: float, k: int):
    """
    Estimate number of k-BB molecules:
        ~ n_bbs * (avg_bf)^(k-1)
    Returns (estimate, log10_estimate).
    """
    if k <= 1:
        # degenerate: 1-BB "molecules" are just the individual BBs
        est = float(n_bbs)
        log10_est = math.log10(n_bbs) if n_bbs > 0 else float("-inf")
        return est, log10_est

    # log-space to avoid overflow
    if avg_bf <= 0:
        return 0.0, float("-inf")

    log10_est = math.log10(n_bbs) + (k - 1) * math.log10(avg_bf)
    est = 10**log10_est if log10_est < 308 else float("inf")  # 1e308 ~ double max
    return est, log10_est


def estimate_state_space(dir_path: str, ks: list[int]):
    dir_path = Path(dir_path)

    # ---- load vocabulary ----
    vocab_file = dir_path / "building_blocks.json"
    with vocab_file.open() as f:
        building_blocks = json.load(f)
    bbs = list(building_blocks.keys())
    n_bbs = len(bbs)
    print(f"[INFO] Loaded {n_bbs} building blocks from {vocab_file}")

    # ---- load reactions ----
    reactions_file = dir_path / "reactions.json"
    with reactions_file.open() as f:
        reactions = json.load(f)
    n_rxns = len(reactions)
    print(f"[INFO] Loaded {n_rxns} reactions from {reactions_file}")

    # ---- load compat ----
    compat_path = dir_path / "compatibility.pt"
    compat = torch.load(compat_path)
    assert compat.dim() == 3, f"Expected compat.ndim=3, got {compat.dim()}"
    n_bbs2, n_rxns2, n_centers = compat.shape
    print(f"[INFO] Loaded compat tensor from {compat_path}: shape={compat.shape}")

    if n_bbs2 != n_bbs:
        raise ValueError(f"BB count mismatch: vocab has {n_bbs}, compat has {n_bbs2} in dim 0")
    if n_rxns2 != n_rxns:
        raise ValueError(
            f"Reaction count mismatch: reactions.json has {n_rxns}, compat has {n_rxns2} in dim 1"
        )

    # ---- average branching factor ----
    avg_bf = compute_avg_branching_factor(compat)
    print(f"[INFO] Average branching factor over all (rxn, center): {avg_bf:.4f}")

    # ---- estimates for each k-BB ----
    print("\n[RESULTS] Estimated search-space sizes:")
    total_log10 = None

    for k in ks:
        est, log10_est = estimate_k_bb_states(n_bbs, avg_bf, k)
        if est == float("inf"):
            est_str = "INF (overflow, see log10)"
        else:
            est_str = f"{est:.3e}"

        print(f"  k = {k:2d} BBs:  ~ {est_str}  " f"(log10 |S_{k}-BB| ≈ {log10_est:.2f})")

        # accumulate log10 total safely
        if total_log10 is None:
            total_log10 = log10_est
        else:
            # log-sum-exp in base 10: log10(a + b) = m + log10(10^(x-m) + 10^(y-m))
            m = max(total_log10, log10_est)
            total_log10 = m + math.log10(10 ** (total_log10 - m) + 10 ** (log10_est - m))

    if total_log10 is not None:
        print(f"\n[RESULTS] Total |S| over k={ks}:  " f"log10 |S_total| ≈ {total_log10:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate state-space size from building_blocks.json, reactions.json and compatibility.pt"
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Directory containing building_blocks.json, reactions.json and compatibility.pt",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="List of BB counts k for which to estimate |S_{k-BB}| (default: 3 4 5)",
    )

    args = parser.parse_args()
    estimate_state_space(args.dir, args.ks)
