from syncogen.api.atomics.reshapable import AtomReshapable
import torch


class AtomMask(AtomReshapable):
    def __init__(self, flat_tensor: torch.Tensor, is_batched: bool = False):
        super().__init__(flat_tensor, is_batched)
        # Assert that the tensor is a binary mask (only contains 0s and 1s)
        assert torch.all(
            (flat_tensor == 0) | (flat_tensor == 1)
        ), "AtomMask tensor must be binary (only contain 0s and 1s)"

    def bool(self):
        """Convert the underlying tensor to bool dtype."""
        self.tensor = self.tensor.bool()
        return self
