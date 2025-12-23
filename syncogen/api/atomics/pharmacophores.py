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
        n_subset: int = None,
        pad_to: int = None,
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
                self.coords.dim() == 3 and self.types.dim() == 2 and padding_mask.dim() == 2
            ), "Batched inputs must be [B,P,3], [B,P], [B,P]"
            assert (
                self.coords.shape[:2] == self.types.shape[:2] == padding_mask.shape[:2]
            ), "Batched coords/types/mask must share [B,P]"
            self.padding_mask = padding_mask
            self.types_onehot = torch.stack(
                [pharm_indices_to_onehot(self.types[b]) for b in range(self.types.shape[0])]
            )
            self.batch_size = self.coords.shape[0]
        else:
            # Expect [P, 3], [P], [P]
            self.coords = pharm_coords
            self.types = pharm_types
            assert (
                self.coords.dim() == 2 and self.types.dim() == 1 and padding_mask.dim() == 1
            ), "Unbatched inputs must be [P,3], [P], [P]"
            assert (
                self.coords.shape[0] == self.types.shape[0] == padding_mask.shape[0]
            ), "Unbatched coords/types/mask must share [P]"
            self.padding_mask = padding_mask
            self.types_onehot = pharm_indices_to_onehot(pharm_types)

        if n_subset is not None:
            # Subset to at most n_subset and pad inside subset to ensure uniform shapes
            self.subset(n_subset)

        if pad_to is not None:
            self.pad(pad_to)

    def subset(self, n_subset):
        """Subset to at most n_subset and pad to exactly n_subset for uniform batching."""
        if self.is_batched:
            B = self.coords.shape[0]
            feat_dim = self.coords.shape[-1]
            n_classes = self.types_onehot.shape[-1]
            pos_padded = torch.zeros((B, n_subset, feat_dim), device=self.coords.device)
            types_padded = torch.zeros((B, n_subset, n_classes), device=self.coords.device)
            pharm_padding_mask = torch.zeros((B, n_subset), device=self.coords.device)
            types_indices = torch.zeros(
                (B, n_subset), dtype=self.types.dtype, device=self.types.device
            )  # pad index=0 by default
            # fill padding class in one-hot by default
            types_padded[:, :, -1] = 1
            for b in range(B):
                valid = (self.padding_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
                cur_len = int(valid.numel())
                if cur_len > n_subset:
                    sel = valid[torch.randperm(cur_len, device=self.coords.device)[:n_subset]]
                    pos_sel = self.coords[b][sel]
                    types_idx_sel = self.types[b][sel]
                    types_oh_sel = self.types_onehot[b][sel]
                    fill_len = n_subset
                else:
                    pos_sel = self.coords[b][valid]
                    types_idx_sel = self.types[b][valid]
                    types_oh_sel = self.types_onehot[b][valid]
                    fill_len = cur_len
                pos_padded[b, :fill_len] = pos_sel
                types_indices[b, :fill_len] = types_idx_sel
                types_padded[b, :fill_len] = types_oh_sel
                pharm_padding_mask[b, :fill_len] = 1
            self.coords = pos_padded
            self.types = types_indices
            self.types_onehot = types_padded
            self.padding_mask = pharm_padding_mask
            self.batch_size = B
            return self
        # Unbatched: produce exactly n_subset by padding/truncating
        valid = (self.padding_mask > 0).nonzero(as_tuple=False).squeeze(-1)
        cur_len = int(valid.numel())
        feat_dim = self.coords.shape[-1]
        n_classes = self.types_onehot.shape[-1]
        pos_padded = torch.zeros((n_subset, feat_dim), device=self.coords.device)
        types_padded = torch.zeros((n_subset, n_classes), device=self.coords.device)
        types_idx_padded = torch.zeros(
            (n_subset,), dtype=self.types.dtype, device=self.types.device
        )
        # default to padding class one-hot
        types_padded[:, -1] = 1
        if cur_len > 0:
            if cur_len > n_subset:
                sel = valid[torch.randperm(cur_len, device=self.coords.device)[:n_subset]]
                pos_sel = self.coords[sel]
                types_idx_sel = self.types[sel]
                types_oh_sel = self.types_onehot[sel]
                fill_len = n_subset
            else:
                pos_sel = self.coords[valid]
                types_idx_sel = self.types[valid]
                types_oh_sel = self.types_onehot[valid]
                fill_len = cur_len
            pos_padded[:fill_len] = pos_sel
            types_idx_padded[:fill_len] = types_idx_sel
            types_padded[:fill_len] = types_oh_sel
        self.coords = pos_padded
        self.types = types_idx_padded
        self.types_onehot = types_padded
        self.padding_mask = (torch.arange(n_subset, device=self.coords.device) < cur_len).to(
            self.coords.dtype
        )
        return self

    def pad(self, n_pad):
        """Pad pharmacophores to fixed size n_pad."""
        if self.is_batched:
            B = self.coords.shape[0]
            pos_padded = torch.zeros((B, n_pad, self.coords.shape[-1]), device=self.coords.device)
            types_padded = torch.zeros(
                (B, n_pad, self.types_onehot.shape[-1]), device=self.coords.device
            )
            pharm_padding_mask = torch.zeros((B, n_pad), device=self.coords.device)
            for b in range(B):
                cur_len = int((self.padding_mask[b] > 0).sum().item())
                if cur_len == n_pad:
                    pharm_padding_mask[b] = torch.ones_like(pharm_padding_mask[b])
                    valid = (self.padding_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
                    pos_padded[b] = self.coords[b, valid]
                    types_padded[b] = self.types_onehot[b, valid]
                else:
                    pharm_padding_mask[b, :cur_len] = 1
                    types_padded[b, :, -1] = 1
                    valid = (self.padding_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
                    valid = valid[:cur_len]
                    types_padded[b, :cur_len] = self.types_onehot[b, valid]
                    pos_padded[b, :cur_len] = self.coords[b, valid]
            self.coords = pos_padded
            self.types_onehot = types_padded
            self.padding_mask = pharm_padding_mask
            return self
        # Unbatched
        cur_len = int((self.padding_mask > 0).sum().item())
        if cur_len == n_pad:
            self.padding_mask = torch.ones_like(self.types)
            return self
        pharm_ones = torch.ones_like(self.types)
        pharm_padding_mask = torch.zeros(n_pad)
        pos_padded = torch.zeros((n_pad, self.coords.shape[1]))
        types_padded = torch.zeros((n_pad, self.types_onehot.shape[1]))
        pharm_padding_mask[:cur_len] = 1
        types_padded[:, -1] = 1
        valid = (self.padding_mask > 0).nonzero(as_tuple=False).squeeze(-1)
        valid = valid[:cur_len]
        types_padded[:cur_len] = self.types_onehot[valid]
        pos_padded[:cur_len] = self.coords[valid]
        self.coords = pos_padded
        self.types_onehot = types_padded
        self.padding_mask = pharm_padding_mask
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
