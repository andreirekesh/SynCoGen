import torch
from typing import Optional, Union, Tuple
from syncogen.api.ops.coordinates_ops import (
    center as ops_center,
    random_rotate as ops_random_rotate,
    random_translate as ops_random_translate,
    kabsch_align as ops_kabsch,
)
class Coordinates:
    """Coordinates class for handling molecular coordinates.

    Use factory methods for construction:
        - Coordinates.from_tensor(coordinates, atom_mask)
        - Coordinates.random(shape, atom_mask, device)

    Or use the constructor directly with a coordinates tensor.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        is_batched: Optional[bool] = None,
    ):
        """Initialize from coordinates tensor (canonical constructor).

        Args:
            coordinates: Coordinate tensor, shape (N, 3) or (B, N, 3)
            atom_mask: Optional mask tensor, shape (N,) or (B, N). Defaults to all ones.
            is_batched: Whether this is batched. Inferred from coordinates.dim() if None.
        """
        # Infer batched mode if not specified
        if is_batched is None:
            # 3D tensor with last dim=3 is batched; 2D with last dim=3 is unbatched
            # 4D tensor is batched BB view; 3D with second-to-last != 3 might be BB view
            is_batched = (
                coordinates.dim() >= 3
                and coordinates.shape[-1] == 3
                and coordinates.dim() > 2
            )

        # Initialize atom_mask if not provided
        if atom_mask is None:
            atom_mask = torch.ones(
                coordinates.shape[:-1],
                device=coordinates.device,
                dtype=coordinates.dtype,
            )

        self.tensor = coordinates
        self.is_batched = is_batched

        self.atom_mask = atom_mask.to(device=coordinates.device, dtype=coordinates.dtype)

        if self.is_batched:
            self.batch_size = coordinates.shape[0]
            self.max_atoms = coordinates.shape[1]
            self.n_atoms = (
                self.atom_mask.reshape(self.batch_size, -1).sum(dim=1).long()
            )
        else:
            self.max_atoms = coordinates.shape[0]
            self.n_atoms = int(self.atom_mask.reshape(-1).sum().item())

        # Optional attachments (initialized empty)
        self.has_pharmacophores = False
        self.pharm_coords = None
        self.pharm_padding_mask = None
        self.has_bonds = False
        self.bonds = None
        self.bonds_mask = None

    # ============ Factory Methods ============

    @classmethod
    def from_tensor(
        cls,
        coordinates: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        is_batched: Optional[bool] = None,
    ) -> "Coordinates":
        """Construct from existing coordinates tensor.

        Args:
            coordinates: Coordinate tensor, shape (N, 3) or (B, N, 3)
            atom_mask: Optional mask tensor
            is_batched: Whether batched (inferred if None)
        """
        return cls(coordinates, atom_mask, is_batched)

    @classmethod
    def random(
        cls,
        shape: Union[torch.Size, Tuple[int, ...]],
        atom_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        is_batched: Optional[bool] = None,
    ) -> "Coordinates":
        """Construct with random normal coordinates.

        Args:
            shape: Shape of coordinates tensor, e.g. (N, 3) or (B, N, 3)
            atom_mask: Optional mask tensor. Defaults to all ones.
            device: Target device
            dtype: Data type for coordinates
            is_batched: Whether batched (inferred from shape if None)
        """
        coords = torch.randn(shape, device=device, dtype=dtype)

        if atom_mask is None:
            atom_mask = torch.ones(shape[:-1], device=device, dtype=dtype)

        if not atom_mask.device.type == device:
            atom_mask = atom_mask.to(device=device)
        # Apply mask to zero out invalid positions
        coords = coords * atom_mask.unsqueeze(-1).to(coords.dtype)

        return cls(coords, atom_mask, is_batched)

    @classmethod
    def uniform(
        cls,
        shape: Union[torch.Size, Tuple[int, ...]],
        atom_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        is_batched: Optional[bool] = None,
    ) -> "Coordinates":
        """Construct with all-zero coordinates (a 'uniform' tensor of zeros).

        Args:
            shape: Shape of coordinates tensor, e.g. (N, 3) or (B, N, 3)
            atom_mask: Optional mask tensor. Defaults to all ones.
            device: Target device
            dtype: Data type for coordinates
            is_batched: Whether batched (inferred from shape if None)
        """
        coords = torch.ones(shape, device=device, dtype=dtype)

        if atom_mask is None:
            atom_mask = torch.ones(shape[:-1], device=device, dtype=dtype)

        if not atom_mask.device.type == device:
            atom_mask = atom_mask.to(device=device)

        return cls(coords, atom_mask, is_batched)

    @classmethod
    def zeros(
        cls,
        shape: Union[torch.Size, Tuple[int, ...]],
        atom_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        is_batched: Optional[bool] = None,
    ) -> "Coordinates":
        """Construct with zero coordinates.

        Args:
            shape: Shape of coordinates tensor
            atom_mask: Optional mask tensor
            device: Target device
            dtype: Data type
            is_batched: Whether batched
        """
        coords = torch.zeros(shape, device=device, dtype=dtype)

        if atom_mask is None:
            atom_mask = torch.ones(shape[:-1], device=device, dtype=dtype)

        return cls(coords, atom_mask, is_batched)

    def attach_pharmacophores(
        self, pharm_coords: torch.Tensor, pharm_padding_mask: torch.Tensor
    ):
        """Attach pharmacophores to coordinates with explicit mask.

        - Keeps atoms and pharm as separate tensors internally
        - Combined mask is computed dynamically via property
        """
        device, dtype = self.tensor.device, self.tensor.dtype
        self.pharm_coords = pharm_coords.to(device=device, dtype=dtype)
        self.pharm_padding_mask = pharm_padding_mask.to(
            device=device, dtype=self.atom_mask.dtype
        )
        self.has_pharmacophores = True
        return self

    def attach_bonds(self, bonds: torch.Tensor, bonds_mask: torch.Tensor):
        """Attach bonds to coordinates with explicit mask.

        Args:
            bonds: [..., n_bonds, 3] where each entry is (i_flat, j_flat, bond_type_index)
            bonds_mask: [..., n_bonds] boolean mask of valid bonds
        """
        device = self.tensor.device
        self.bonds = bonds.to(device=device)
        self.bonds_mask = bonds_mask.to(device=device, dtype=torch.bool)
        self.has_bonds = True
        return self

    @property
    def atom_coords(self):
        """Get atom coordinates tensor."""
        return self.tensor

    @property
    def pharmacophores(self):
        """Get pharmacophore coordinates (everything after max_atoms)."""
        if not self.has_pharmacophores:
            raise ValueError(
                "Cannot access pharmacophores: this Coordinates object does not have pharmacophores"
            )
        return self.pharm_coords

    def apply_mask(self, mask: torch.Tensor):
        """Apply a mask to the coordinates tensor.

        Args:
            mask: Tensor of shape matching self.tensor up to the last dimension
        """
        assert (
            mask.shape == self.tensor.shape[:-1]
        ), f"Mask shape {mask.shape} must match tensor shape {self.tensor.shape[:-1]}"
        self.tensor = self.tensor * mask.unsqueeze(-1)
        return self

    def get_center(self, custom_mask=None):
        """Get the center of mass according to atom mask."""
        if custom_mask is not None:
            assert (
                custom_mask.shape == self.tensor.shape[:-1]
            ), f"Mask shape {custom_mask.shape} must match tensor shape {self.tensor.shape[:-1]}"
            mask = custom_mask.unsqueeze(-1)
        else:
            assert (
                self.atom_mask is not None
            ), "Must provide atom_mask to compute center"
            mask = self.atom_mask.unsqueeze(-1).bool()
        if self.is_batched:
            centers = (self.tensor * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            return centers
        center = (self.tensor * mask).sum(dim=0) / (mask.sum() + 1e-8)
        return center

    def center(self, custom_mask=None, custom_center=None):
        """Center using atom mask only; move pharmacophores along if attached."""
        if custom_mask is not None:
            assert (
                custom_mask.shape == self.tensor.shape[:-1]
            ), f"Mask shape {custom_mask.shape} must match tensor shape {self.tensor.shape[:-1]}"
            mask = custom_mask.bool()
        else:
            assert (
                self.atom_mask is not None
            ), "Must provide atom_mask when no custom center is given"
            mask = self.atom_mask.bool()
        if not self.has_pharmacophores:
            self.tensor = ops_center(self.tensor, mask, custom_center)
            return self
        # Concatenate atoms + pharm; compute center from atoms only
        coords_cat = torch.cat(
            [self.tensor, self.pharm_coords], dim=1 if self.is_batched else 0
        )
        zeros_pharm = torch.zeros_like(
            self.pharm_padding_mask, dtype=mask.dtype, device=self.tensor.device
        )
        compute_mask = torch.cat([mask, zeros_pharm], dim=1 if self.is_batched else 0)
        # apply_mask: atoms use mask, pharms use pharm_padding_mask (keep valid pharms, zero padding)
        apply_mask = torch.cat(
            [mask, self.pharm_padding_mask.bool()], dim=1 if self.is_batched else 0
        )
        out = ops_center(coords_cat, compute_mask, custom_center, apply_mask=apply_mask)
        # Split back
        if self.is_batched:
            self.tensor = out[:, : self.max_atoms]
            self.pharm_coords = out[:, self.max_atoms :]
        else:
            self.tensor = out[: self.max_atoms]
            self.pharm_coords = out[self.max_atoms :]
        return self

    def is_centered(self, tol=1e-6):
        """Check if coordinates are centered according to atom mask."""
        centers = self.get_center()
        return torch.all(torch.abs(centers) < tol)

    def scale(self, multiplier):
        """Scale coordinates (and pharmacophores if attached) by a multiplier."""
        assert multiplier > 0, "Scale multiplier must be greater than 0"
        self.tensor = self.tensor * multiplier
        if self.has_pharmacophores and self.pharm_coords is not None:
            self.pharm_coords = self.pharm_coords * multiplier
        return self

    def random_rotate(self):
        """Apply random rotation to coordinates (and pharmacophores if attached)."""
        if not self.has_pharmacophores:
            self.tensor = ops_random_rotate(self.tensor, self.atom_mask)
            return self
        coords_cat = torch.cat(
            [self.tensor, self.pharm_coords], dim=1 if self.is_batched else 0
        )
        out = ops_random_rotate(coords_cat, self.atom_and_pharmacophore_mask)
        if self.is_batched:
            self.tensor = out[:, : self.max_atoms]
            self.pharm_coords = out[:, self.max_atoms :]
        else:
            self.tensor = out[: self.max_atoms]
            self.pharm_coords = out[self.max_atoms :]
        return self

    def random_translate(self, scale: float = 1.0):
        """Apply random translation to coordinates (and pharmacophores if attached)."""
        if not self.has_pharmacophores:
            self.tensor = ops_random_translate(
                self.tensor, self.atom_mask, scale
            )
            return self
        coords_cat = torch.cat(
            [self.tensor, self.pharm_coords], dim=1 if self.is_batched else 0
        )
        out = ops_random_translate(coords_cat, self.atom_and_pharmacophore_mask, scale)
        if self.is_batched:
            self.tensor = out[:, : self.max_atoms]
            self.pharm_coords = out[:, self.max_atoms :]
        else:
            self.tensor = out[: self.max_atoms]
            self.pharm_coords = out[self.max_atoms :]
        return self

    def kabsch_align_to(
        self,
        reference: torch.Tensor,
        mask: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        """Align (rotate) coordinates to reference using Kabsch (weighted rigid align).

        Args:
            reference: Tensor of shape [N, 3] to align to
            mask: Optional mask of shape [N] for valid points
            weights: Optional weights of shape [N]
        Returns:
            self
        """
        if not self.has_pharmacophores:
            use_mask = self.atom_mask if mask is None else mask
            self.tensor = ops_kabsch(self.tensor, reference, use_mask, weights)
            return self
        # Concatenate coords and build masks/weights
        coords_cat = torch.cat(
            [self.tensor, self.pharm_coords], dim=1 if self.is_batched else 0
        )
        # Reference is atoms-only; pad to match concatenated length so pharm points "ride along" with zero weight
        if self.is_batched:
            B = reference.shape[0]
            pharm_len = self.pharm_coords.shape[1]
            pad = torch.zeros(
                (B, pharm_len, 3), device=reference.device, dtype=reference.dtype
            )
            reference_cat = torch.cat([reference, pad], dim=1)
            apply_mask = self.atom_and_pharmacophore_mask
            w = torch.zeros_like(apply_mask, dtype=coords_cat.dtype)
            w[:, : self.max_atoms] = 1
        else:
            pharm_len = self.pharm_coords.shape[0]
            pad = torch.zeros(
                (pharm_len, 3), device=reference.device, dtype=reference.dtype
            )
            reference_cat = torch.cat([reference, pad], dim=0)
            apply_mask = self.atom_and_pharmacophore_mask
            w = torch.zeros_like(apply_mask, dtype=coords_cat.dtype)
            w[: self.max_atoms] = 1
        out = ops_kabsch(coords_cat, reference_cat, apply_mask, w)
        if self.is_batched:
            self.tensor = out[:, : self.max_atoms]
            self.pharm_coords = out[:, self.max_atoms :]
        else:
            self.tensor = out[: self.max_atoms]
            self.pharm_coords = out[self.max_atoms :]
        return self

    def set_mask(self, atom_mask: torch.Tensor, apply_mask: bool = True):
        """Set a new atom mask; must match existing mask shape. Recomputes n_atoms and re-applies zeroing."""
        assert (
            atom_mask.shape == self.atom_mask.shape
        ), f"New mask shape {atom_mask.shape} must match existing mask shape {self.atom_mask.shape}"
        self.atom_mask = atom_mask.long()
        # Update n_atoms derived from mask
        if self.is_batched:
            assert hasattr(
                self, "batch_size"
            ), "batch_size must be set for batched Coordinates"
            self.n_atoms = (
                self.atom_mask.reshape(self.batch_size, -1).sum(dim=1).long()
            )
        else:
            self.n_atoms = int(self.atom_mask.reshape(-1).sum().item())
        # Maintain invariant: masked positions are zero
        self.tensor = self.tensor * self.atom_mask.unsqueeze(-1).to(
            self.tensor.dtype
        )
        return self

    def set_coordinates(self, coordinates: torch.Tensor, apply_mask: bool = True):
        """Replace coordinates tensor; must match existing tensor shape. Optionally re-apply mask zeroing."""
        assert (
            coordinates.shape == self.tensor.shape
        ), f"New coordinates shape {coordinates.shape} must match existing shape {self.tensor.shape}"
        self.tensor = coordinates
        if apply_mask:
            self.tensor = self.tensor * self.atom_mask.unsqueeze(-1).to(
                self.tensor.dtype
            )
        return self

    def to_numpy(self):
        """Convert coordinates to numpy array."""
        return self.tensor.detach().cpu().numpy()

    def to_torch(self):
        """Convert coordinates to torch tensor."""
        if not torch.is_tensor(self.tensor):
            self.tensor = torch.from_numpy(self.tensor)
        return self

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """Move coordinates to specified device and/or dtype.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
            dtype: Target dtype (e.g., torch.float32)
        """
        if device is not None:
            self.tensor = self.tensor.to(device)
            if self.atom_mask is not None:
                self.atom_mask = self.atom_mask.to(device)
            if self.pharm_padding_mask is not None:
                self.pharm_padding_mask = self.pharm_padding_mask.to(device)
            if self.pharm_coords is not None:
                self.pharm_coords = self.pharm_coords.to(device)
            if self.bonds is not None:
                self.bonds = self.bonds.to(device)
            if self.bonds_mask is not None:
                self.bonds_mask = self.bonds_mask.to(device)

        if dtype is not None:
            self.tensor = self.tensor.to(dtype)
            if self.pharm_coords is not None:
                self.pharm_coords = self.pharm_coords.to(dtype)

        return self

    def clone(self) -> "Coordinates":
        """Create a deep copy of this Coordinates object."""
        coords = Coordinates(
            coordinates=self.tensor.clone(),
            atom_mask=self.atom_mask.clone(),
            is_batched=self.is_batched,
        )
        if self.has_pharmacophores:
            coords.attach_pharmacophores(
                pharm_coords=(
                    self.pharm_coords.clone() if self.pharm_coords is not None else None
                ),
                pharm_padding_mask=(
                    self.pharm_padding_mask.clone()
                    if self.pharm_padding_mask is not None
                    else None
                ),
            )
        if self.has_bonds:
            coords.attach_bonds(
                bonds=self.bonds.clone() if self.bonds is not None else None,
                bonds_mask=(
                    self.bonds_mask.clone() if self.bonds_mask is not None else None
                ),
            )
        return coords

    def __len__(self):
        if not self.is_batched:
            raise TypeError("len() is only valid for batched Coordinates")
        return self.batch_size

    def __getitem__(self, idx):
        if not self.is_batched:
            raise TypeError("Indexing is only valid for batched Coordinates")
        coords = Coordinates(
            coordinates=self.tensor[idx],
            atom_mask=(
                self.atom_mask[idx] if self.atom_mask is not None else None
            ),
            is_batched=False,
        )
        if self.has_pharmacophores:
            coords.attach_pharmacophores(
                pharm_coords=(
                    self.pharm_coords[idx] if self.pharm_coords is not None else None
                ),
                pharm_padding_mask=(
                    self.pharm_padding_mask[idx]
                    if self.pharm_padding_mask is not None
                    else None
                ),
            )
        if self.has_bonds:
            coords.attach_bonds(
                bonds=self.bonds[idx] if self.bonds is not None else None,
                bonds_mask=(
                    self.bonds_mask[idx] if self.bonds_mask is not None else None
                ),
            )
        return coords

    @property
    def dtype(self):
        """Get dtype of coordinates."""
        return self.tensor.dtype

    @property
    def device(self):
        """Get device of coordinates."""
        return self.tensor.device

    @property
    def shape(self):
        """Get shape of coordinates."""
        return self.tensor.shape

    @property
    def atom_and_pharmacophore_mask(self):
        """Dynamically compute concatenated atom + pharmacophore mask in atoms view."""
        m = self.atom_mask
        if self.is_batched:
            # Flatten any BB dims into atoms view
            m_atoms = m.reshape(m.shape[0], -1)
            if self.has_pharmacophores and self.pharm_padding_mask is not None:
                return torch.cat([m_atoms, self.pharm_padding_mask], dim=1)
            return m_atoms
        # Unbatched
        m_atoms = m.reshape(-1)
        if self.has_pharmacophores and self.pharm_padding_mask is not None:
            return torch.cat([m_atoms, self.pharm_padding_mask], dim=0)
        return m_atoms
