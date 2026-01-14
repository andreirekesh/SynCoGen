#!/usr/bin/env python
"""Sampling script for SynCoGen diffusion model.

Usage:
    python sample.py --config configs/experiments/default.gin \
        --vocab_dir vocabulary/original \
        --checkpoint_path path/to/checkpoint.ckpt \
        --output_dir samples/run1 \
        --num_batches 10

With pharmacophore conditioning:
    python sample.py --config configs/experiments/default.gin \
        --vocab_dir vocabulary/original \
        --checkpoint_path path/to/checkpoint.ckpt \
        --output_dir samples/run1 \
        --reference_ligand path/to/ligand.sdf \
        --num_batches 10
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import gin
import torch
from rdkit import Chem

from syncogen.constants.constants import load_vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from SynCoGen diffusion model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to gin config file"
    )
    parser.add_argument(
        "--vocab_dir", type=str, required=True, help="Path to vocabulary directory"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save sampled molecules",
    )
    parser.add_argument(
        "--reference_ligand",
        type=str,
        default=None,
        help="Reference ligand SDF for conditioning",
    )
    parser.add_argument(
        "--num_batches", type=int, default=10, help="Number of batches to sample"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--sort_by", type=str, choices=["energy", "similarity"], default=None
    )
    parser.add_argument(
        "--metadata_file", type=str, default=None, help="Path to save metadata JSON"
    )
    parser.add_argument(
        "--gin", type=str, action="append", default=[], help="Additional gin bindings"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pharmacophore_conditioning",
        action="store_true",
        help="Enable pharmacophore conditioning by setting SyncogenDataManager.load_pharmacophores=True in gin before Diffusion is instantiated",
    )
    return parser.parse_args()


# Parse args early to load vocabulary before other imports
args = parse_args()
load_vocabulary(args.vocab_dir)

from syncogen.diffusion.training.diffusion import Diffusion
from syncogen.data.dataloader import SyncogenDataManager
from syncogen.utils.rdkit import (
    calc_energy,
    save_as_sdf,
    mol_to_pharm_cond,
    set_mol_coordinates,
)
import syncogen.logging.loggers
import syncogen.logging.callbacks
from syncogen.diffusion.training.trainer import Trainer


@gin.configurable
def sample(
    checkpoint_path: str,
    output_dir: str,
    num_batches: int = 10,
    num_steps: int = 100,
    reference_ligand: Optional[str] = None,
    sort_by: Optional[str] = None,
    metadata_file: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    device: str = "cuda",
    seed: int = 42,
):
    """Main sampling function."""

    import lightning as L

    L.seed_everything(seed, workers=True)

    device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create data manager
    data_manager = SyncogenDataManager()
    if batch_size_override is not None:
        data_manager.eval_batch_size = batch_size_override

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = Diffusion.load_from_checkpoint(
        checkpoint_path,
        data_manager=data_manager,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Pharmacophore conditioning
    pharm_cond = None
    ref_mol = None
    if reference_ligand is not None:
        print(f"Loading reference ligand from {reference_ligand}")
        ref_mol = Chem.MolFromMolFile(reference_ligand)
        if ref_mol is None:
            raise ValueError(f"Could not load reference ligand from {reference_ligand}")

        types, pos, mask = mol_to_pharm_cond(
            ref_mol,
            batch_size=model.eval_batch_size,
            n_subset=model.pharm_subset,
            center=True,
            normalize=True,
        )
        pharm_cond = (types.to(device), pos.to(device), mask.to(device))

    results = []
    total_sampled = 0
    total_valid = 0

    print(f"Sampling {num_batches} batches of {model.eval_batch_size} molecules...")

    for batch_idx in range(num_batches):
        with torch.no_grad():
            # Use the model's built-in method with optional direct conditioning
            graphs, coords = model.restore_model_and_sample(
                num_steps=num_steps,
                cond=pharm_cond,
            )

        batch_size = graphs.batch_size
        total_sampled += batch_size

        for i in range(batch_size):
            graph_i = graphs[i]
            coords_i = coords[i]

            # Only check unmasked status for valid nodes (not padding)
            n = int(graph_i.lengths.item())
            if not (
                graph_i.unmasked_bbs[:n].all() and graph_i.unmasked_rxns[:n, :n].all()
            ):
                continue

            try:
                mol = graph_i.build_rdkit(return_smiles=False)
            except Exception as e:
                print(f"Failed to build molecule: {e}")
                continue

            if mol is None:
                continue

            total_valid += 1

            atom_coords = coords_i.atom_coords.reshape(-1, 3)
            atom_mask = graph_i.ground_truth_atom_mask.tensor.reshape(-1).bool()
            valid_coords = atom_coords[: atom_mask.shape[0], :][atom_mask]
            mol = set_mol_coordinates(mol, valid_coords.cpu())

            energy = calc_energy(mol)
            similarity = _compute_similarity(mol, ref_mol) if ref_mol else None

            smiles = Chem.MolToSmiles(mol)
            bb_indices = graph_i.bb_indices[: int(graph_i.lengths.item())].tolist()
            rxn_tuple = graph_i.rxn_tuple

            results.append(
                {
                    "idx": len(results),
                    "smiles": smiles,
                    "energy": energy,
                    "similarity": similarity,
                    "building_blocks": bb_indices,
                    "reactions": [r.tolist() for r in rxn_tuple],
                    "mol": mol,
                }
            )

        print(
            f"Batch {batch_idx + 1}/{num_batches}: {total_valid}/{total_sampled} valid"
        )

    # Sort
    if sort_by == "energy":
        results = sorted(
            results,
            key=lambda x: x["energy"] if x["energy"] is not None else float("inf"),
        )
    elif sort_by == "similarity" and ref_mol is not None:
        results = sorted(results, key=lambda x: -(x["similarity"] or 0))

    # Save SDFs
    print(f"\nSaving {len(results)} molecules to {output_path}")
    for i, r in enumerate(results):
        props = {"energy_kcal_mol": r["energy"], "smiles": r["smiles"]}
        if r["similarity"] is not None:
            props["similarity"] = r["similarity"]
        save_as_sdf(r["mol"], str(output_path / f"mol_{i:04d}.sdf"), properties=props)

    # Save metadata
    p_valid = total_valid / total_sampled if total_sampled > 0 else 0
    metadata = {
        "total_sampled": total_sampled,
        "total_valid": total_valid,
        "p_valid": p_valid,
        "molecules": [{k: v for k, v in r.items() if k != "mol"} for r in results],
    }

    metadata_path = (
        Path(metadata_file) if metadata_file else output_path / "metadata.json"
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    energies = [r["energy"] for r in results if r["energy"] is not None]
    print(f"\n{'='*60}")
    print(f"Sampling complete!")
    print(f"  Valid molecules: {total_valid}/{total_sampled} ({p_valid:.1%})")
    if energies:
        print(
            f"  Energy (kcal/mol): mean={sum(energies)/len(energies):.1f}, "
            f"median={sorted(energies)[len(energies)//2]:.1f}"
        )
    if ref_mol:
        sims = [r["similarity"] for r in results if r["similarity"] is not None]
        if sims:
            print(f"  Similarity: mean={sum(sims)/len(sims):.3f}, max={max(sims):.3f}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    return results


def _compute_similarity(mol: Chem.Mol, ref_mol: Chem.Mol) -> Optional[float]:
    """Compute shape similarity between generated and reference molecule."""
    try:
        from shepherd_score.alignment import crippen_align
        from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist

        mol_aligned = crippen_align(ref_mol, mol)
        dist = ShapeTanimotoDist(mol_aligned, ref_mol)
        return 1.0 - dist
    except Exception:
        return None


def main():
    gin_bindings = list(args.gin)  # make a copy
    if args.pharmacophore_conditioning:
        gin_bindings.append("SyncogenDataManager.load_pharmacophores = True")
    else:
        gin_bindings.append("SyncogenDataManager.load_pharmacophores = False")

    gin.parse_config_files_and_bindings(
        config_files=[args.config],
        bindings=gin_bindings,
    )

    sample(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        num_steps=args.num_steps,
        reference_ligand=args.reference_ligand,
        sort_by=args.sort_by,
        metadata_file=args.metadata_file,
        batch_size_override=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
