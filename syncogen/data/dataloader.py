import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import gin
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch
from syncogen.utils.file_readers import (
    get_coordinates,
    get_pharmacophores,
    select_conformer_key,
)
from syncogen.constants.constants import MAX_ATOMS_PER_BB
from syncogen.api.graph.graph import BBRxnGraph
from syncogen.data.utils import to_dense_graph


@gin.configurable
class SyncogenDataManager:
    def __init__(
        self,
        *,
        graphs_path: Union[str, Path],
        conformers_path: Optional[Union[str, Path]] = None,
        pharmacophore_path: Optional[Union[str, Path]] = None,
        train_size: float = 0.9,
        validation_size: float = 0.1,
        test_size: float = 0.0,
        overfit: bool = False,
        n_overfit: Optional[int] = None,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        sample_conformer: bool = False,
        load_pharmacophores: bool = False,
        load_bonds: bool = False,
        coord_mask_value: float = 0.0,
        valid_seed: Optional[int] = None,
        max_bbs=5,
    ):
        self.graphs_path = Path(graphs_path)
        self.conformers_path = (
            None if conformers_path is None else Path(conformers_path)
        )
        self.pharmacophore_path = (
            None if pharmacophore_path is None else Path(pharmacophore_path)
        )
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.overfit = overfit
        self.n_overfit = n_overfit
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.sample_conformer = sample_conformer
        self.load_pharmacophores = load_pharmacophores
        self.load_bonds = load_bonds
        self.coord_mask_value = coord_mask_value
        self.valid_seed = valid_seed
        self._splits_cache: Optional[Dict[str, List[Data]]] = None
        self.max_bbs = max_bbs
        self.max_atoms = max_bbs * MAX_ATOMS_PER_BB
        self.train_length_values: Optional[torch.Tensor] = None
        self.train_length_probs: Optional[torch.Tensor] = None

    def get_split_cache_dir(self) -> Path:
        extension = f"/overfit_{self.n_overfit}" if self.overfit else "/full"
        cache_dir_splits = self.graphs_path.parent / ("splits" + extension)
        return cache_dir_splits

    def get_split_path(self, split_name: str) -> Path:
        return self.get_split_cache_dir() / f"{split_name}.pt"

    def get_lengths_path(self) -> Path:
        return self.get_split_cache_dir() / "train_lengths_categorical.json"

    def get_graph_data_splits(self) -> Dict[str, List[Data]]:
        import copy

        split_names = ["train", "validation", "test"]
        cache_dir_splits = self.get_split_cache_dir()
        cache_dir_splits.mkdir(parents=True, exist_ok=True)

        if not all(self.get_split_path(split).exists() for split in split_names):
            data_list: List[Data] = torch.load(self.graphs_path)
            if self.overfit and self.n_overfit is not None:
                if self.n_overfit == 1:  # single example
                    data_list = [data_list[0] for _ in range(100)]
                else:
                    data_list = data_list[: self.n_overfit]

            if self.overfit:
                train_data = data_list
                splits: Dict[str, List[Data]] = {
                    "train": train_data,
                    "validation": train_data,
                    "test": [],
                }
            else:
                total_size = len(data_list)
                train_end = int(total_size * self.train_size)
                val_end = train_end + int(total_size * self.validation_size)
                test_end = val_end + int(total_size * self.test_size)

                splits: Dict[str, List[Data]] = {
                    "train": data_list[:train_end],
                    "validation": data_list[train_end:val_end],
                    "test": data_list[val_end:test_end],
                }

            for split, data in splits.items():
                torch.save(data, self.get_split_path(split))
        else:
            splits = {
                split: torch.load(self.get_split_path(split)) for split in split_names
            }

        self._splits_cache = splits
        self._load_or_compute_train_lengths()
        return splits

    def _load_or_compute_train_lengths(self):
        lengths_path = self.get_lengths_path()
        if lengths_path.exists():
            with open(lengths_path) as f:
                payload = json.load(f)
            lengths = torch.tensor([int(k) for k in payload.keys()], dtype=torch.long)
            probs = torch.tensor(
                [float(v) for v in payload.values()], dtype=torch.float
            )
        else:
            # Only allow explicit compute if splits cache is present
            # (previous logic, not used unless called via data splits creation)
            if self._splits_cache is None or "train" not in self._splits_cache:
                raise FileNotFoundError(
                    f"Length file {lengths_path} does not exist and cannot compute lengths without train split present."
                )
            train_list = self._splits_cache["train"]
            lengths_raw = [
                int(getattr(d, "num_nodes", d.x.shape[0])) for d in train_list
            ]
            lengths_tensor = torch.as_tensor(lengths_raw, dtype=torch.long)
            values, counts = lengths_tensor.unique(return_counts=True)
            probs = counts.float() / counts.sum()
            lengths = values
            payload = {
                int(l): float(p) for l, p in zip(lengths.tolist(), probs.tolist())
            }
            with open(lengths_path, "w") as f:
                json.dump(payload, f)
        self.train_length_values = lengths
        self.train_length_probs = probs

    def ensure_train_lengths_loaded(self):
        """
        Loads the train lengths/probs from the lengths_path.
        Raises if path not present.
        """
        lengths_path = self.get_lengths_path()
        if not lengths_path.exists():
            raise FileNotFoundError(
                f"Expected train lengths file at {lengths_path}, but could not find it. Please generate the splits/lengths first."
            )
        with open(lengths_path) as f:
            payload = json.load(f)
        self.train_length_values = torch.tensor(
            [int(k) for k in payload.keys()], dtype=torch.long
        )
        self.train_length_probs = torch.tensor(
            [float(v) for v in payload.values()], dtype=torch.float
        )

    def sample_n_nodes(self, batch_size: int) -> Optional[torch.Tensor]:
        # Ensure train length probabilities are loaded
        if self.train_length_probs is None or self.train_length_values is None:
            self.ensure_train_lengths_loaded()
        idx = torch.multinomial(
            self.train_length_probs, num_samples=batch_size, replacement=True
        )
        return self.train_length_values[idx]

    def _make_datasets(self) -> Tuple[Dataset, Dataset]:
        if self.conformers_path is None:
            raise ValueError(
                "conformers_path must be provided to build datasets and dataloaders."
            )
        splits = self._splits_cache or self.get_graph_data_splits()
        train_ds = SyncogenDataset(
            conformers_path=str(self.conformers_path),
            data_list=splits["train"],
            sample_conformer=self.sample_conformer,
            coord_mask_value=self.coord_mask_value,
            pharmacophore_path=self.pharmacophore_path,
            load_pharmacophores=self.load_pharmacophores,
            load_bonds=self.load_bonds,
        )
        valid_ds = SyncogenDataset(
            conformers_path=str(self.conformers_path),
            data_list=splits["validation"],
            sample_conformer=self.sample_conformer,
            coord_mask_value=self.coord_mask_value,
            pharmacophore_path=self.pharmacophore_path,
            load_pharmacophores=self.load_pharmacophores,
            load_bonds=self.load_bonds,
        )
        return train_ds, valid_ds

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds, valid_ds = self._make_datasets()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=self.num_workers > 0,
        )

        if self.valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(self.valid_seed)

        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
        )
        return train_loader, valid_loader


@gin.configurable
class SyncogenDataset(Dataset):
    def __init__(
        self,
        conformers_path: str,
        data_list: List[Data],
        pharmacophore_path: Optional[str] = None,
        sample_conformer: bool = False,
        coord_mask_value: float = 0.0,
        load_pharmacophores: bool = False,
        load_bonds: bool = False,
    ):
        super(SyncogenDataset, self).__init__()
        self.data_list = data_list
        self.pharmacophore_path = pharmacophore_path
        self.coord_mask_value = coord_mask_value
        self.sample_conformer = sample_conformer
        self.conformers_path = conformers_path
        self.load_pharmacophores = load_pharmacophores
        self.load_bonds = load_bonds

    def len(self):
        return len(self.data_list)

    def get(self, idx: int):
        data = self.data_list[idx]
        # Pick a conformer
        if self.sample_conformer:
            key = select_conformer_key(data.data_index, Path(self.conformers_path))
        else:
            key = f"mol_{data.data_index}_final_conf_0"

        # Compute ground truth atom mask from graph to account for dropped atoms
        # Convert sparse graph to dense format (single graph, so batch size = 1)
        batch = torch.zeros(
            data.x.shape[0], dtype=torch.long
        )  # All nodes in same batch
        X_dense, E_dense, _, _ = to_dense_graph(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=batch,
            max_num_nodes=data.x.shape[0],  # Use actual number of nodes
        )
        # Remove batch dimension (single graph)
        X_dense = X_dense[0]
        E_dense = E_dense[0]

        # Create graph and get ground truth atom mask
        # Single graph without padding - all nodes are valid
        n_nodes = X_dense.shape[0]
        node_mask = torch.ones(n_nodes, dtype=torch.bool, device=X_dense.device)
        graph = BBRxnGraph.from_onehot(X_dense, E_dense, node_mask=node_mask)
        atom_mask = (
            graph.ground_truth_atom_mask.bool()
        )  # Shape: [n_fragments * MAX_ATOMS_PER_BB], True = valid
        # Reshape to [n_fragments, MAX_ATOMS_PER_BB]
        n_fragments = X_dense.shape[0]
        atom_mask = atom_mask.reshape(n_fragments, MAX_ATOMS_PER_BB)

        coordinates, coords_mask, bonds = get_coordinates(
            key,
            Path(self.conformers_path),
            mask_value=self.coord_mask_value,
            return_bonds=self.load_bonds,
            atom_mask=atom_mask,
        )

        if self.load_pharmacophores:
            pharm_types, pharm_pos, pharm_vectors = get_pharmacophores(
                key,
                Path(self.pharmacophore_path),
            )
            data.pharm_types = pharm_types
            data.pharm_pos = pharm_pos
            # Vectors currently unused downstream; stored for completeness if needed later
            data.pharm_vectors = pharm_vectors
            # Per-graph pharmacophore count so batching can derive pharm_batch later
            data.pharm_len = torch.tensor(len(pharm_types), dtype=torch.long)

        if self.load_bonds:
            data.bonds = bonds
            data.bonds_len = torch.tensor(len(bonds), dtype=torch.long)

        data.coordinates = coordinates
        data.coords_mask = coords_mask

        return data


########################################################
# Fault-tolerant dataloaders from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
########################################################


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):
    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop("shuffle", None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.generator.get_state(), "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
