import math

import torch
import torch.nn as nn
from torch.nn import init

from syncogen.constants.constants import MAX_ATOMS_PER_BB


class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        """SE(3)-equivariant MLP layer for coordinate following:
        PosMLP(R) = \prod^{CoM}(MLP(||R||)\frac{R}{||R||+\delta})
        """
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        """Args:
            pos: torch.Tensor, shape [bs, n, 3]
            node_mask: torch.Tensor, shape [bs, n]
        Returns:
            new_pos: torch.Tensor, shape [bs, n, 3]
        """
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model

        # Precompute sinusoidal embeddings
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # Frequencies

        pe = torch.zeros(max_len, d_model)
        # Ensure div_term has correct size for both sin and cos
        n_dims_even = math.ceil(pe.size(1) / 2)
        n_dims_odd = pe.size(1) - n_dims_even

        pe[:, 0::2] = torch.sin(position * div_term[:n_dims_even])  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term[:n_dims_odd])  # Odd indices: cos

        self.register_buffer("pe", pe)  # Save embeddings as a buffer (non-trainable)

    def forward(self, positions):
        return self.pe[positions].to(positions.device)


class HierarchicalPositionalEmbedding(nn.Module):
    """
    Creates positional embeddings that respect the hierarchical structure of molecules:
    fragments containing atoms.
    """

    def __init__(self, d_model, max_fragments=5, max_atoms_per_fragment=MAX_ATOMS_PER_BB):
        super().__init__()
        self.d_model = d_model
        self.max_fragments = max_fragments
        self.max_atoms = max_atoms_per_fragment

        # Split the embedding dimension between fragment and atom positions
        self.fragment_dim = d_model // 2
        self.atom_dim = d_model - self.fragment_dim

        # Create separate positional embeddings for fragments and atoms
        self.fragment_embedding = SinusoidalPositionalEmbedding(
            d_model=self.fragment_dim, max_len=max_fragments
        )

        self.atom_embedding = SinusoidalPositionalEmbedding(
            d_model=self.atom_dim, max_len=max_atoms_per_fragment
        )

    def forward(self, fragment_features, atom_features=None):
        """
        Args:
            fragment_features: Fragment-level features tensor [batch_size, n_fragments, feature_dim]
            atom_features: Optional atom-level features tensor [batch_size, n_fragments*MAX_ATOMS, feature_dim]

        Returns:
            Tuple of (fragment_pe, atom_pe) where each is positional embeddings matching input shapes
        """
        # Calculate fragment indices from fragment features shape
        batch_size, n_fragments = fragment_features.shape[:2]
        fragment_indices = torch.arange(n_fragments, device=fragment_features.device)
        fragment_indices = fragment_indices.expand(batch_size, -1)

        # Get fragment embeddings
        frag_pe = self.fragment_embedding(fragment_indices)

        # If no atom features, return fragment embeddings with zero atom embeddings
        padding = torch.zeros(
            *frag_pe.shape[:-1], self.atom_dim, device=frag_pe.device, dtype=frag_pe.dtype
        )

        if atom_features is None:
            return torch.cat([frag_pe, padding], dim=-1), None

        # Calculate atom indices from atom features shape
        atom_indices = torch.arange(self.max_atoms, device=atom_features.device)
        atom_indices = atom_indices.expand(batch_size, n_fragments, -1)
        atom_indices = atom_indices.reshape(batch_size, -1)

        # Get atom embeddings
        atom_pe = self.atom_embedding(atom_indices)

        # Expand fragment embeddings to match atom shape
        frag_pe_expanded = frag_pe.unsqueeze(-2).expand(-1, -1, self.max_atoms, -1)
        frag_pe_expanded = frag_pe_expanded.reshape(batch_size, -1, self.fragment_dim)

        # Combine fragment and atom embeddings for atom-level features
        atom_full_pe = torch.cat([frag_pe_expanded, atom_pe], dim=-1)

        # Return both embeddings
        return torch.cat([frag_pe, padding], dim=-1), atom_full_pe


class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D or 2-D Tensor of indices. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: a Tensor of positional embeddings with shape (..., D).
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        # Reshape t to (..., 1) and freqs to (1, ..., half)
        t_shape = list(t.shape) + [1]
        freqs_shape = [1] * len(t.shape) + [half]

        args = t.view(*t_shape).float() * freqs.view(*freqs_shape)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SimplePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len, d_model):
        super().__init__()
        # Create parameter with dim number of max_len dimensions
        shape = (max_len,) * dim + (
            d_model,
        )  # e.g., (max_len, d_model) or (max_len, max_len, d_model)
        self.pe = nn.Parameter(torch.randn(*shape))

    def forward(self, x):
        # Slice the parameter to match input size
        slice_indices = tuple(slice(0, x.size(i + 1)) for i in range(len(x.shape) - 2))
        return x + self.pe[slice_indices]


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Maps node features to global features using Principal Neighborhood Aggregation.
        PNA(X)_i = W^T [mean, min, max, std]_j(x_ij) where j \in N(i)
        """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """Args:
            X: bs, n, dx. (node features with dimension dx)
            x_mask: bs, n. (node mask)
        Returns:
            out: bs, dy. (global feature with dimension dy)
        """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features using Principal Neighborhood Aggregation."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """Args:
            E: bs, n, n, de (Pairwise edge feature matrix with dimension de)
            e_mask1: bs, n, 1, 1
            e_mask2: bs, 1, n, 1
        Returns:
            out: bs, dy (Global feature with dimension dy)
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])  # bs, n, n, de
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class EtoX(nn.Module):
    """Map edge features to node features using Principal Neighborhood Aggregation."""

    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """Args:
            E: bs, n, n, de
            e_mask2: bs, n, n
        Returns:
            out: bs, n, dx
        """
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


class CtoE(nn.Module):
    """Map coordinate features to edge features using Principal Neighborhood Aggregation."""

    def __init__(self, dc, de):
        super().__init__()
        self.lin = nn.Linear(4 * dc, de)
        self.eps = 1e-8

    def forward(self, delta1, c_mask1, c_mask2):
        """Args:
            delta1: bs, nm, nm, dc
            c_mask1: bs, nm, 1, 1
            c_mask2: bs, 1, nm, 1
        Returns:
            out: bs, n, n, de
        """
        bs, nm, _, dc = delta1.shape
        assert nm % MAX_ATOMS_PER_BB == 0
        n = nm // MAX_ATOMS_PER_BB

        mask = (c_mask1 * c_mask2).expand(-1, -1, -1, dc)  # bs, nm, nm, dc
        mask = mask.view(bs, n, MAX_ATOMS_PER_BB, n, MAX_ATOMS_PER_BB, dc)  # bs, n, m, n, m, dc
        delta1 = delta1.view(bs, n, MAX_ATOMS_PER_BB, n, MAX_ATOMS_PER_BB, dc)  # bs, n, m, n, m, dc

        float_imask = 1 - mask.float()  # bs, n, m, n, m, dc
        divide = torch.sum(mask, dim=(2, 4)) + self.eps  # bs, n, n, dc

        m = delta1.sum(dim=(2, 4)) / divide  # bs, n, n, dc
        mi = (delta1 + 1e5 * float_imask).min(dim=4)[0].min(dim=2)[0]  # bs, n, n, dc
        ma = (delta1 - 1e5 * float_imask).max(dim=4)[0].max(dim=2)[0]  # bs, n, n, dc
        Z = (delta1 - m[:, :, None, :, None, :]) ** 2  # bs, n, m, n, m, dc
        std = torch.sum(Z * mask, dim=(2, 4)) / divide  # bs, n, n, dc
        z = torch.cat((m, mi, ma, std), dim=-1)  # bs, n, n, 4 * dc
        out = self.lin(z)  # bs, n, n, de
        return out


class CtoX(nn.Module):
    """Map coordinate features to node features using Principal Neighborhood Aggregation."""

    def __init__(self, dc, dx):
        super().__init__()
        self.c_to_e = CtoE(dc, dc)  # bs, nm, nm, dc -> bs, n, n, dc
        self.e_to_x = EtoX(dc, dx)  # bs, n, n, dc -> bs, n, dx

    def forward(self, delta1, c_mask1, c_mask2, e_mask2):
        """Args:
            delta1: bs, n*m, n*m, dc
            pos_mask: bs, n*m, n*m
            e_mask2: bs, n, n
        Returns:
            out: bs, n, dx
        """
        E = self.c_to_e(delta1, c_mask1, c_mask2)  # bs, n, n, dc
        return self.e_to_x(E, e_mask2)  # bs, n, dx


class Ctoy(nn.Module):
    """Map coordinate features to global features using Principal Neighborhood Aggregation."""

    def __init__(self, dc, dy):
        super().__init__()
        self.c_to_e = CtoE(dc, dc)  # bs, nm, nm, dc -> bs, n, n, dc
        self.e_to_y = Etoy(dc, dy)  # bs, n, n, dc -> bs, dy

    def forward(self, delta1, c_mask1, c_mask2, e_mask1, e_mask2):
        """Args:
            delta1: bs, n*m, n*m, dc
            pos_mask: bs, n*m, n*m
            e_mask2: bs, n, n
        Returns:
            out: bs, dy
        """
        E = self.c_to_e(delta1, c_mask1, c_mask2)  # bs, n, n, dc
        return self.e_to_y(E, e_mask1, e_mask2)  # bs, dy


class EtoC(nn.Module):
    """Broadcast edge features to coordinate space and apply a linear layer."""

    def __init__(self, dy, dc):
        super().__init__()
        self.lin = nn.Linear(dy, dc, bias=False)
        self.pe_m = SinusoidalPositionalEmbedding(d_model=dy, max_len=MAX_ATOMS_PER_BB**2)

    def forward(self, E, e_mask1, e_mask2):
        """Args:
            E: bs, n, n, dy
            e_mask1: bs, n, 1, 1
            e_mask2: bs, 1, n, 1
        Returns:
            out: bs, nm, nm, dc
        """
        bs, n, _, dy = E.shape
        nm = n * MAX_ATOMS_PER_BB

        # Step 1: Broadcast masks to shape (bs, nm, nm, 1)
        mask1 = e_mask1.expand(bs, n, MAX_ATOMS_PER_BB, 1).reshape(
            bs, nm, 1, 1
        )  # Expand and reshape for first axis
        mask2 = e_mask2.expand(bs, MAX_ATOMS_PER_BB, n, 1).reshape(
            bs, 1, nm, 1
        )  # Expand and reshape for second axis
        mask = mask1 * mask2

        # Broadcast edge features to the coords space (broadcast E to (bs, nm, nm, dy))
        E_broadcast = E.unsqueeze(2).unsqueeze(4)  # (bs, n, 1, n, 1, dy)
        E_broadcast = E_broadcast.expand(
            bs, n, MAX_ATOMS_PER_BB, n, MAX_ATOMS_PER_BB, dy
        )  # (bs, n, m, n, m, dy)

        # Add position embeddings to each m x m block of E_broadcast
        pos = torch.arange(MAX_ATOMS_PER_BB**2, device=E.device).reshape(
            MAX_ATOMS_PER_BB, MAX_ATOMS_PER_BB
        )  # (m, m)
        pos_emb = self.pe_m(pos).unsqueeze(1).unsqueeze(0).unsqueeze(0)  # (1, 1, m, 1, m, dy)
        E_broadcast = E_broadcast + pos_emb  # (bs, n, m, n, m, dy)

        # Combine the broadcast edge features with masks
        E_masked = E_broadcast.reshape(bs, nm, nm, dy) * mask  # (bs, nm, nm, dy)

        # Step 2: Apply the linear layer to project to coordinate features
        out = self.lin(E_masked)  # (bs, nm, nm, dc)

        return out


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """SE(3)-equivariant layer normalization for coordinates:
        Note: There is a relatively similar layer implemented by NVIDIA:
        https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
        It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1,)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        node_mask = node_mask.unsqueeze(-1)
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(
            node_mask, dim=1, keepdim=True
        )  # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)


class SetNorm(nn.LayerNorm):
    """SetNorm: A Layer Normalization for Set Functions over unordered nodes"""

    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, x, x_mask):
        bs, n, d = x.shape
        divide = torch.sum(x_mask, dim=1, keepdim=True) * d  # bs
        means = torch.sum(x * x_mask, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((x - means) ** 2 * x_mask, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (x - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * x_mask
        return out


class GraphNorm(nn.LayerNorm):
    """GraphNorm: A Layer Normalization for Graph Functions over pairwise edge matrix"""

    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, E, emask1, emask2):
        bs, n, _, d = E.shape
        divide = torch.sum(emask1 * emask2, dim=[1, 2], keepdim=True) * d  # bs
        means = torch.sum(E * emask1 * emask2, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((E - means) ** 2 * emask1 * emask2, dim=[1, 2], keepdim=True) / (
            divide + self.eps
        )
        out = (E - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * emask1 * emask2
        return out
