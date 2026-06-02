"""Synthesizability scoring: SA score plus optional retrosynthesis success.

The two retrosynthesis back-ends are optional and disabled by default:

* `AiZynthFinder <https://github.com/MolecularAI/aizynthfinder>`_
* `Syntheseus <https://github.com/microsoft/syntheseus>`_
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig

# SA_Score ships in the RDKit Contrib directory, which is not importable by default.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
except ImportError:
    AiZynthFinder = None

try:
    from syntheseus import Molecule
    from syntheseus.search.mol_inventory import SmilesListInventory
    from syntheseus.reaction_prediction.inference import LocalRetroModel, RootAlignedModel
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch, MolIsPurchasableCost
    from syntheseus.search.node_evaluation.common import ReactionModelLogProbCost, ConstantNodeEvaluator
    from syntheseus.search.analysis.route_extraction import iter_routes_time_order
    _HAS_SYNTHESEUS = True
except ImportError:
    _HAS_SYNTHESEUS = False

_BH2 = Chem.MolFromSmarts("[B;H2]")
_BORONIC = Chem.MolFromSmiles("B(O)O")


def _bh2_to_boronic(mol: Chem.Mol) -> Chem.Mol:
    """Replace BH2 with B(OH)2 so retrosynthesis tools accept the boron."""
    if not mol.HasSubstructMatch(_BH2):
        return mol
    out = Chem.ReplaceSubstructs(mol, _BH2, _BORONIC, replaceAll=True)[0]
    Chem.SanitizeMol(out)
    return out


def _select_all(collection: Any) -> List[str]:
    """Select every stock/policy defined in the AiZynthFinder config (no hard-coded names)."""
    keys = list(getattr(collection, "items", []) or [])
    if not keys:
        return []
    try:
        collection.select(keys if len(keys) > 1 else keys[0])
    except Exception:
        collection.select(keys[0])
        keys = [keys[0]]
    return keys


class Synthesizability:
    """Compute SA score and, optionally, retrosynthesis success per molecule."""

    def __init__(
        self,
        *,
        aizynth: bool = False,
        aizynth_config: Optional[Union[str, Path]] = None,
        syntheseus: bool = False,
        syntheseus_inventory: Optional[Iterable[str]] = None,
        syntheseus_model: str = "local_retro",
        time_limit_s: int = 60,
    ) -> None:
        RDLogger.DisableLog("rdApp.*")

        self.aizynth = aizynth
        if aizynth:
            if AiZynthFinder is None:
                raise ImportError("aizynthfinder is not installed but aizynth=True.")
            if aizynth_config is None:
                raise ValueError("aizynth=True requires aizynth_config.")
            self._finder = AiZynthFinder(configfile=str(aizynth_config))
            _select_all(self._finder.stock)
            _select_all(self._finder.expansion_policy)
            _select_all(self._finder.filter_policy)

        self.syntheseus = syntheseus
        if syntheseus:
            if not _HAS_SYNTHESEUS:
                raise ImportError("syntheseus is not installed but syntheseus=True.")
            self._inventory = SmilesListInventory(smiles_list=list(syntheseus_inventory or []))
            model = LocalRetroModel if syntheseus_model.lower() == "local_retro" else RootAlignedModel
            self._search = RetroStarSearch(
                reaction_model=model(use_cache=True, default_num_results=10),
                mol_inventory=self._inventory,
                or_node_cost_fn=MolIsPurchasableCost(),
                and_node_cost_fn=ReactionModelLogProbCost(normalize=False),
                value_function=ConstantNodeEvaluator(0.0),
                time_limit_s=time_limit_s,
            )

    def _aizynth_one(self, smi: str):
        if not self.aizynth:
            return None, None
        self._finder.target_smiles = smi
        self._finder.tree_search()
        self._finder.build_routes()
        stats = self._finder.extract_statistics()
        return stats["is_solved"], stats["top_score"]

    def _syntheseus_one(self, smi: str) -> Optional[bool]:
        if not self.syntheseus:
            return None
        self._search.reset()
        try:
            graph, _ = self._search.run_from_mol(Molecule(smi))
            return len(list(iter_routes_time_order(graph, max_routes=5))) > 0
        except Exception as exc:
            logger.warning("Syntheseus failed for %s: %s", smi, exc)
            return False

    def __call__(self, mols: Iterable[Chem.Mol]) -> List[Dict[str, Any]]:
        from tqdm import tqdm

        results = []
        for mol in tqdm(mols, desc="Synthesizability"):
            mol = _bh2_to_boronic(Chem.RemoveHs(mol))
            smi = Chem.MolToSmiles(mol, canonical=True)
            solved, score = self._aizynth_one(smi)
            results.append({
                "sa_score": float(sascorer.calculateScore(mol)),
                "aizynth_solved": solved,
                "aizynth_score": score,
                "syntheseus_solved": self._syntheseus_one(smi),
            })
        return results
