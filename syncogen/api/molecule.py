from typing import Optional, List
import gin
import torch

from syncogen.api.graph.graph import BBRxnGraph
from syncogen.api.atomics.coordinates import Coordinates
from syncogen.api.atomics.pharmacophores import ShepherdPharmacophores


@gin.configurable
class SyncogenMolecule:
    """Molecule class for handling molecular data.

    Supports loading from SMILES, RDKit mol object, and coordinates.
    Handles batch operations and coordinate transformations.
    """

    def __init__(
        self,
        graph: BBRxnGraph = None,
        smiles: str = None,
        coords: Coordinates = None,
        pharm: ShepherdPharmacophores = None,
        bonds: Optional[torch.Tensor] = None,
        # Only for noisy initialization
        length: int = None,
        n_atoms: int = None,
        is_batched: bool = False,
        batch_size: int = None,
    ):
        self.is_batched = bool(
            is_batched or (graph is not None and getattr(graph, "is_batched", False))
        )
        self.graph = graph
        self.smiles = smiles
        self.coords = coords
        self.pharm = pharm
        self.bonds = bonds

        if graph is None:
            if self.is_batched:
                if batch_size is None:
                    raise ValueError(
                        "batch_size required for batched SyncogenMolecule initialization"
                    )
                self.graph = BBRxnGraph.masked(
                    max_nodes=length, batch_size=batch_size, n_nodes=[length] * batch_size
                )
                self.coords = Coordinates(
                    n_atoms=n_atoms, max_atoms=n_atoms, batch_size=batch_size, is_batched=True
                )
            else:
                self.graph = BBRxnGraph.masked(max_nodes=length, n_nodes=length)
                self.coords = Coordinates(n_atoms=n_atoms, max_atoms=n_atoms)

    def reconstruct_smiles(self) -> str:
        return self.smiles

    def set_coords(self, coords: Coordinates):
        self.coords = coords
        return self

    def set_bonds(self, bonds: torch.Tensor):
        self.bonds = bonds
        return self

    def set_smiles(self, smiles: str):
        self.smiles = smiles
        return self

    def set_pharmacophores(
        self,
        pharm_coords: torch.Tensor,
        pharm_types: torch.Tensor,
        n_subset: int = None,
        append_to_coords: bool = False,
    ):
        """Set pharmacophores from provided data.

        Args:
            pharm_coords: Pharmacophore coordinates tensor
            pharm_types: Pharmacophore types tensor
            n_subset: If provided and len(types) > n_subset, randomly subset pharmacophores
            append_to_coords: If True, append pharmacophores to coordinates
        """
        self.pharm = ShepherdPharmacophores(
            pharm_coords=pharm_coords,
            pharm_types=pharm_types,
            n_subset=n_subset,
            is_batched=self.is_batched,
        )

        if self.coords is not None and append_to_coords:
            self.coords.append_pharmacophores(self.pharm.coords)

        return self

    def __len__(self):
        if not self.is_batched:
            raise TypeError("len() is only valid for batched SyncogenMolecule")
        return self.graph.batch_size

    def __getitem__(self, idx):
        if not self.is_batched:
            raise TypeError("Indexing is only valid for batched SyncogenMolecule")
        return SyncogenMolecule(
            graph=self.graph[idx],
            smiles=(
                self.smiles[idx]
                if isinstance(self.smiles, list) and idx < len(self.smiles)
                else None
            ),
            coords=self.coords[idx] if self.coords is not None else None,
            bonds=(
                self.bonds[idx] if isinstance(self.bonds, list) and idx < len(self.bonds) else None
            ),
            pharm=self.pharm[idx] if self.pharm is not None else None,
            is_batched=False,
        )
