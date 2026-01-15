from syncogen.constants.constants import MAX_ATOMS_PER_BB
import torch


class AtomReshapable:
    def __init__(self, flat_tensor: torch.Tensor, is_batched: bool = False):
        """Reshapable base with simple view detection and reshaping.

        Accepts either atoms view ([N,...]/[B,N,...]) or bbs view ([n_bb,MA,...]/[B,n_bb,MA,...]).
        Keeps only one current view (no caching of the other).
        """
        self.is_batched = is_batched
        self.tensor = flat_tensor
        atom_dim = 1 if self.is_batched else 0
        # Validate shape consistency without storing mutable view state
        if not (
            flat_tensor.dim() > atom_dim + 1
            and flat_tensor.shape[atom_dim + 1] == MAX_ATOMS_PER_BB
        ):
            n_along = flat_tensor.shape[atom_dim]
            assert (
                n_along % MAX_ATOMS_PER_BB == 0
            ), f"Atom dimension must be divisible by MAX_ATOMS_PER_BB, got {n_along}"

    def reshape_to_atoms(self):
        """Reshape from BB view to atoms view if needed."""
        if not self.is_bbs_view:
            return self
        t = self.tensor
        if self.is_batched:
            batch_size = t.shape[0]
            nbb = t.shape[1]
            ma = t.shape[2]
            remaining_dims = t.shape[3:]
            self.tensor = t.reshape(batch_size, nbb * ma, *remaining_dims)
        else:
            nbb = t.shape[0]
            ma = t.shape[1]
            remaining_dims = t.shape[2:]
            self.tensor = t.reshape(nbb * ma, *remaining_dims)
        return self

    def reshape_to_bbs(self):
        """Reshape from atoms view to BB view if divisible."""
        if self.is_bbs_view:
            return self
        t = self.tensor
        if self.is_batched:
            batch_size = t.shape[0]
            n = t.shape[1]
            assert (
                n % MAX_ATOMS_PER_BB == 0
            ), "Number of atoms N must be divisible by MAX_ATOMS_PER_BB to reshape to (n_bb, MA)"
            nbb = n // MAX_ATOMS_PER_BB
            remaining_dims = t.shape[2:]
            self.tensor = t.reshape(batch_size, nbb, MAX_ATOMS_PER_BB, *remaining_dims)
        else:
            n = t.shape[0]
            assert (
                n % MAX_ATOMS_PER_BB == 0
            ), "Number of atoms N must be divisible by MAX_ATOMS_PER_BB to reshape to (n_bb, MA)"
            nbb = n // MAX_ATOMS_PER_BB
            remaining_dims = t.shape[1:]
            self.tensor = t.reshape(nbb, MAX_ATOMS_PER_BB, *remaining_dims)
        return self

    @property
    def is_bbs_view(self) -> bool:
        atom_dim = 1 if self.is_batched else 0
        return (
            self.tensor.dim() > atom_dim + 1
            and self.tensor.shape[atom_dim + 1] == MAX_ATOMS_PER_BB
        )

    @property
    def n_building_blocks(self) -> int:
        atom_dim = 1 if self.is_batched else 0
        if self.is_bbs_view:
            return self.tensor.shape[atom_dim]
        n_along = self.tensor.shape[atom_dim]
        assert (
            n_along % MAX_ATOMS_PER_BB == 0
        ), "Atom dimension must be divisible by MAX_ATOMS_PER_BB"
        return n_along // MAX_ATOMS_PER_BB

    def to(self, device=None, dtype=None):
        """Convert tensor to specified device and/or dtype."""
        if device is not None:
            self.tensor = self.tensor.to(device)
        if dtype is not None:
            self.tensor = self.tensor.to(dtype)
        return self

    def cpu(self):
        """Move tensor to CPU."""
        self.tensor = self.tensor.cpu()
        return self

    def cuda(self, device=None):
        """Move tensor to CUDA device."""
        self.tensor = self.tensor.cuda(device)
        return self

    def float(self):
        """Convert tensor to float32."""
        self.tensor = self.tensor.float()
        return self

    def double(self):
        """Convert tensor to float64."""
        self.tensor = self.tensor.double()
        return self

    def half(self):
        """Convert tensor to float16."""
        self.tensor = self.tensor.half()
        return self

    def int(self):
        """Convert tensor to int32."""
        self.tensor = self.tensor.int()
        return self

    def long(self):
        """Convert tensor to int64."""
        self.tensor = self.tensor.long()
        return self
