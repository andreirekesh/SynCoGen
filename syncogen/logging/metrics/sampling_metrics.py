from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Sequence

import gin
import numpy as np
from rdkit import Chem

from syncogen.api.graph.graph import BBRxnGraph
from syncogen.utils.rdkit import is_valid_smiles, calc_energy


class MetricsBase(ABC):
    """Base class for all sampling metrics."""

    @abstractmethod
    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        """Compute metrics from graphs and pre-built molecules.

        Args:
            graphs: BBRxnGraph object (batched or unbatched)
            mols: List of RDKit molecules (None for failed reconstructions)
            **kwargs: Additional metric-specific arguments

        Returns:
            Dictionary of metric names to values
        """
        pass


@gin.configurable
class MetricsList(MetricsBase):
    def __init__(self, metrics: Sequence[MetricsBase] = ()):
        self.metrics = metrics

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(graphs, mols, **kwargs))
        return results


@gin.configurable
class ValidityMetrics(MetricsBase):
    """Metrics related to molecular validity."""

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        n_total = len(mols)
        if n_total == 0:
            return {}

        valid_mols = [m for m in mols if m is not None and is_valid_smiles(m)]
        n_valid = len(valid_mols)

        return {
            "sampling/p_valid": n_valid / n_total,
            "sampling/n_valid": float(n_valid),
            "sampling/n_total": float(n_total),
        }


@gin.configurable
class UniquenessMetrics(MetricsBase):
    """Metrics related to molecular uniqueness."""

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        n_total = len(mols)
        if n_total == 0:
            return {}

        valid_smiles = [
            Chem.MolToSmiles(m) for m in mols if m is not None and is_valid_smiles(m)
        ]
        n_valid = len(valid_smiles)

        if n_valid == 0:
            return {}

        unique_valid = set(valid_smiles)
        return {
            "sampling/uniqueness": len(unique_valid) / n_valid,
            "sampling/n_unique": float(len(unique_valid)),
        }


@gin.configurable
class NoveltyMetrics(MetricsBase):
    """Metrics related to molecular novelty."""

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        train_smiles: Optional[Set[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if train_smiles is None:
            return {}

        n_total = len(mols)
        if n_total == 0:
            return {}

        valid_smiles = [
            Chem.MolToSmiles(m) for m in mols if m is not None and is_valid_smiles(m)
        ]
        n_valid = len(valid_smiles)

        if n_valid == 0:
            return {}

        n_novel = len([s for s in valid_smiles if s not in train_smiles])
        return {
            "sampling/novelty": n_novel / n_total,
            "sampling/n_novel": float(n_novel),
        }


@gin.configurable
class FragmentUsageMetrics(MetricsBase):
    """Metrics related to fragment (building block) usage."""

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        if not graphs.is_batched:
            lengths = graphs.lengths
            bb_indices = graphs.bb_indices[: lengths[0]]
            arr = bb_indices.cpu().numpy()
            per_graph_counts = [len(arr)]
        else:
            lengths = graphs.lengths
            parts = []
            for b in range(graphs.batch_size):
                n = int(lengths[b].item())
                parts.append(graphs.bb_indices[b, :n])
            arr = np.concatenate([x.cpu().numpy() for x in parts])
            per_graph_counts = [int(l.item()) for l in lengths]

        unique = np.unique(arr)
        return {
            "sampling/n_unique_fragments": float(len(unique)),
            "sampling/n_total_fragments": float(len(arr)),
            "sampling/avg_fragments_per_mol": float(np.mean(per_graph_counts)),
        }


@gin.configurable
class EnergyMetrics(MetricsBase):
    """Metrics related to molecular energy."""

    def __init__(self, per_atom: bool = False):
        self.per_atom = per_atom

    def compute(
        self,
        graphs: BBRxnGraph,
        mols: List[Optional[Chem.Mol]],
        **kwargs,
    ) -> Dict[str, float]:
        energies = []
        for mol in mols:
            if mol is None:
                continue
            energy = calc_energy(mol, per_atom=self.per_atom)
            if energy is not None:
                energies.append(energy)

        if not energies:
            return {}

        energies = np.array(energies)
        suffix = "_per_atom" if self.per_atom else ""
        return {
            f"sampling/energy{suffix}_mean": float(np.mean(energies)),
            f"sampling/energy{suffix}_median": float(np.median(energies)),
            "sampling/n_energy_computed": float(len(energies)),
        }
