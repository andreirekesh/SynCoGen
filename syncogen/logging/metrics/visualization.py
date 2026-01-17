from __future__ import annotations

from typing import List, Optional

import numpy as np
import wandb
from rdkit import Chem

from syncogen.api.graph.graph import BBRxnGraph


def log_molecule_images(
    run: "wandb.sdk.wandb_run.Run",
    graphs: BBRxnGraph,
    mols: List[Optional[Chem.Mol]],
    step: int,
    max_mols: int = 4,
    prefix: str = "molecule_images",
) -> None:
    """Log a few RDKit molecule images to wandb."""
    logged = 0
    used_indices = []
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
            run.log(
                {
                    f"{prefix}/step_{step}_idx_{i}": wandb.Molecule.from_rdkit(
                        mol,
                        convert_to_3d_and_optimize=False,
                        caption=smiles,
                    )
                }
            )
            logged += 1
            used_indices.append(i)
        except Exception:
            continue
        if logged >= max_mols:
            break


def log_fragment_histogram(
    run: "wandb.sdk.wandb_run.Run",
    graphs: BBRxnGraph,
    step: int,
    prefix: str = "fragment_distributions",
) -> None:
    """Log a histogram of fragment indices used in the batch."""
    if not graphs.is_batched:
        lengths = graphs.lengths
        bb = graphs.bb_indices[: lengths[0]]
        all_indices = bb.cpu().numpy()
    else:
        lengths = graphs.lengths
        parts = []
        for b in range(graphs.batch_size):
            n = int(lengths[b].item())
            parts.append(graphs.bb_indices[b, :n].cpu().numpy())
        all_indices = np.concatenate(parts)

    if all_indices.size == 0:
        return

    unique_indices, counts = np.unique(all_indices, return_counts=True)
    table_data = [[int(idx), int(count)] for idx, count in zip(unique_indices, counts)]
    table = wandb.Table(data=table_data, columns=["Fragment Index", "Count"])

    run.log(
        {
            f"{prefix}/step_{step}": wandb.plot.bar(
                table,
                "Fragment Index",
                "Count",
                title=f"Fragment Distribution at Step {step}",
            )
        }
    )
