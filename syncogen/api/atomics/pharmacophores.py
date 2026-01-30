import torch
import gin
from syncogen.constants.constants import N_PHARM


def pharm_indices_to_onehot(indices):
    """Convert pharmacophore type indices to one-hot encoding."""
    # Assuming N_PHARM pharmacophore types (0-N_PHARM-1) plus padding class
    n_classes = N_PHARM + 1
    onehot = torch.zeros((indices.shape[0], n_classes), device=indices.device)
    onehot.scatter_(1, indices.unsqueeze(1), 1)
    return onehot


class ShepherdPharmacophores:
    """Pharmacophores with unified batched/unbatched API."""

    def __init__(
        self,
        pharm_coords,
        pharm_types,
        padding_mask,
        min_subset: int = None,
        max_subset: int = None,
        is_batched: bool = False,
    ):
        self.is_batched = bool(
            is_batched or (torch.is_tensor(pharm_coords) and pharm_coords.dim() == 3)
        )
        if self.is_batched:
            # Expect [B, P, 3], [B, P], [B, P]
            self.coords = pharm_coords
            self.types = pharm_types
            assert (
                self.coords.dim() == 3
                and self.types.dim() == 2
                and padding_mask.dim() == 2
            ), "Batched inputs must be [B,P,3], [B,P], [B,P]"
            assert (
                self.coords.shape[:2] == self.types.shape[:2] == padding_mask.shape[:2]
            ), "Batched coords/types/mask must share [B,P]"
            self.padding_mask = padding_mask
            self.types_onehot = torch.stack(
                [
                    pharm_indices_to_onehot(self.types[b])
                    for b in range(self.types.shape[0])
                ]
            )
            self.batch_size = self.coords.shape[0]
        else:
            # Expect [P, 3], [P], [P]
            self.coords = pharm_coords
            self.types = pharm_types
            assert (
                self.coords.dim() == 2
                and self.types.dim() == 1
                and padding_mask.dim() == 1
            ), "Unbatched inputs must be [P,3], [P], [P]"
            assert (
                self.coords.shape[0] == self.types.shape[0] == padding_mask.shape[0]
            ), "Unbatched coords/types/mask must share [P]"
            self.padding_mask = padding_mask
            self.types_onehot = pharm_indices_to_onehot(pharm_types)

        if min_subset is not None and max_subset is not None:
            # Subset with random selection between min_subset and max_subset
            self.subset_with_range(min_subset, max_subset)

    def _subset_single_sample(
        self,
        valid_indices,
        coords,
        types,
        types_onehot,
        min_subset: int,
        max_subset: int,
    ):
        """Subset a single sample: randomly select n_subset between min_subset and max_subset.

        If cur_len > n_subset: randomly select n_subset pharmacophores.
        If cur_len <= n_subset: shuffle and take all.

        Returns:
            pos_sel: Selected coordinates [fill_len, 3]
            types_idx_sel: Selected type indices [fill_len]
            types_oh_sel: Selected type one-hot [fill_len, n_classes]
            fill_len: Number of valid pharmacophores selected
        """
        cur_len = int(valid_indices.numel())

        # Randomly select n_subset for this sample between min_subset and max_subset
        n_subset = torch.randint(
            min_subset, max_subset + 1, (1,), device=coords.device
        ).item()

        if cur_len > n_subset:
            # Randomly select n_subset pharmacophores
            sel = valid_indices[
                torch.randperm(cur_len, device=coords.device)[:n_subset]
            ]
            fill_len = n_subset
        else:
            # Shuffle even when taking all to prevent positional bias
            sel = valid_indices[torch.randperm(cur_len, device=coords.device)]
            fill_len = cur_len

        pos_sel = coords[sel]
        types_idx_sel = types[sel]
        types_oh_sel = types_onehot[sel]

        return pos_sel, types_idx_sel, types_oh_sel, fill_len

    def subset_with_range(self, min_subset: int, max_subset: int):
        """Apply subsetting to all samples and pad to max_subset for uniform batching."""
        assert min_subset <= max_subset, "min_subset must be <= max_subset"

        feat_dim = self.coords.shape[-1]
        n_classes = self.types_onehot.shape[-1]

        if self.is_batched:
            B = self.coords.shape[0]
            # Pad to max_subset for uniform batching
            pos_padded = torch.zeros(
                (B, max_subset, feat_dim), device=self.coords.device
            )
            types_padded = torch.zeros(
                (B, max_subset, n_classes), device=self.coords.device
            )
            pharm_padding_mask = torch.zeros((B, max_subset), device=self.coords.device)
            types_indices = torch.zeros(
                (B, max_subset), dtype=self.types.dtype, device=self.types.device
            )
            # fill padding class in one-hot by default
            types_padded[:, :, -1] = 1

            for b in range(B):
                valid = (self.padding_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
                pos_sel, types_idx_sel, types_oh_sel, fill_len = (
                    self._subset_single_sample(
                        valid,
                        self.coords[b],
                        self.types[b],
                        self.types_onehot[b],
                        min_subset,
                        max_subset,
                    )
                )
                pos_padded[b, :fill_len] = pos_sel
                types_indices[b, :fill_len] = types_idx_sel
                types_padded[b, :fill_len] = types_oh_sel
                pharm_padding_mask[b, :fill_len] = 1

            self.coords = pos_padded
            self.types = types_indices
            self.types_onehot = types_padded
            self.padding_mask = pharm_padding_mask
            self.batch_size = B
        else:
            # Unbatched: randomly select n_subset between min_subset and max_subset
            valid = (self.padding_mask > 0).nonzero(as_tuple=False).squeeze(-1)
            pos_padded = torch.zeros((max_subset, feat_dim), device=self.coords.device)
            types_padded = torch.zeros(
                (max_subset, n_classes), device=self.coords.device
            )
            types_idx_padded = torch.zeros(
                (max_subset,), dtype=self.types.dtype, device=self.types.device
            )
            # default to padding class one-hot
            types_padded[:, -1] = 1

            pos_sel, types_idx_sel, types_oh_sel, fill_len = self._subset_single_sample(
                valid,
                self.coords,
                self.types,
                self.types_onehot,
                min_subset,
                max_subset,
            )
            pos_padded[:fill_len] = pos_sel
            types_idx_padded[:fill_len] = types_idx_sel
            types_padded[:fill_len] = types_oh_sel

            self.coords = pos_padded
            self.types = types_idx_padded
            self.types_onehot = types_padded
            self.padding_mask = (
                torch.arange(max_subset, device=self.coords.device) < fill_len
            ).to(self.coords.dtype)

        return self

    def __len__(self):
        if not self.is_batched:
            raise TypeError("len() is only valid for batched ShepherdPharmacophores")
        return self.batch_size

    def __getitem__(self, idx):
        if not self.is_batched:
            raise TypeError("Indexing is only valid for batched ShepherdPharmacophores")
        return ShepherdPharmacophores(
            pharm_coords=self.coords[idx],
            pharm_types=self.types[idx],
            padding_mask=self.padding_mask[idx],
            is_batched=False,
        )
