import torch


def _ensure_time_dim(t: torch.Tensor, target_dim: int) -> torch.Tensor:
    while t.dim() < target_dim:
        t = t.unsqueeze(-1)
    return t


def center(
    coords: torch.Tensor,
    mask: torch.Tensor,
    custom_center: torch.Tensor = None,
    apply_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Center coordinates by mask; supports [N,3], [B,N,3], or [B,S,N,3].

    Args:
        coords: Coordinates tensor [N,3], [B,N,3], or [B,S,N,3]
        mask: Mask for computing center
        custom_center: Optional pre-computed center to subtract
        apply_mask: Optional separate mask for zeroing after centering.
                    If None, NO zeroing is applied (matches old behavior).
    """
    if custom_center is not None:
        centered = coords - custom_center
        if apply_mask is not None:
            return centered * apply_mask.unsqueeze(-1)
        return centered

    # Use dim=-2 to always sum over the nodes dimension (second to last)
    # This works for [N,3], [B,N,3], and [B,S,N,3]
    c = (coords * mask.unsqueeze(-1)).sum(dim=-2, keepdim=True) / (
        mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-8
    )
    centered = coords - c
    if apply_mask is not None:
        return centered * apply_mask.unsqueeze(-1)
    return centered


@torch.amp.autocast(device_type="cuda", enabled=False)
def random_rotate(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply random rotation(s) to coords, preserving masked-out positions."""
    orig_dtype = coords.dtype
    coords = coords.to(torch.float32)

    if coords.dim() == 2:
        q, r = torch.linalg.qr(
            torch.randn(3, 3, device=coords.device, dtype=torch.float32)
        )
        R = q @ torch.diag(torch.sign(torch.diag(r)))
        if torch.det(R) < 0:
            R[:, -1] *= -1
        out = coords @ R.T
        return (out * mask.unsqueeze(-1)).to(orig_dtype)
    b = coords.shape[0]
    R_list = []
    for _ in range(b):
        q, r = torch.linalg.qr(
            torch.randn(3, 3, device=coords.device, dtype=torch.float32)
        )
        R = q @ torch.diag(torch.sign(torch.diag(r)))
        if torch.det(R) < 0:
            R[:, -1] *= -1
        R_list.append(R)
    R = torch.stack(R_list, dim=0)
    out = torch.einsum("bij,bkj->bik", coords, R)
    return (out * mask.unsqueeze(-1)).to(orig_dtype)


def random_translate(
    coords: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Apply random translation(s) to coords, preserving masked-out positions."""
    if coords.dim() == 2:
        t = torch.randn(1, 3, device=coords.device) * scale
        return (coords + t) * mask.unsqueeze(-1)
    t = torch.randn(coords.shape[0], 1, 3, device=coords.device) * scale
    return (coords + t) * mask.unsqueeze(-1)


@torch.amp.autocast(device_type="cuda", enabled=False)
def kabsch_align(
    src: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor = None,
    debug: bool = True,
) -> torch.Tensor:
    """Align src to ref using Kabsch; supports [N,3] or [B,N,3]."""
    orig_dtype = src.dtype

    single_input = src.dim() == 2
    if single_input:
        src, ref, mask = src.unsqueeze(0), ref.unsqueeze(0), mask.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)

    # Cast to float32 for numerical stability
    src = src.to(torch.float32)
    ref = ref.to(torch.float32)

    m = mask.to(dtype=torch.float32, device=src.device)
    w = (
        torch.ones_like(m, device=src.device, dtype=torch.float32)
        if weights is None
        else weights.to(torch.float32)
    )
    w = w * m
    wexp = w.unsqueeze(-1).expand_as(src)

    # 1. Centroids
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_cent = (ref * wexp).sum(dim=1, keepdim=True) / wsum.unsqueeze(-1)
    src_cent = (src * wexp).sum(dim=1, keepdim=True) / wsum.unsqueeze(-1)

    ref_c = ref - ref_cent
    src_c = src - src_cent

    # 2. Covariance
    # H = src_centered^T @ ref_centered (to align src to ref)
    cov = torch.einsum("bij,bik->bjk", (wexp * src_c), ref_c)

    # 3. SVD
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    V = Vh.mH

    # 4. Rotation + reflection correction
    R = torch.einsum("bij,bkj->bik", U, V)
    detR = torch.det(R)
    F = (
        torch.eye(3, dtype=torch.float32, device=cov.device)
        .unsqueeze(0)
        .repeat(cov.shape[0], 1, 1)
    )
    F[:, -1, -1] = detR
    R = torch.einsum("bij,bjk,blk->bil", U, F, V)

    # 5. Apply rotation + translation
    aligned = (src_c @ R) + ref_cent
    aligned = aligned.to(orig_dtype)
    return aligned.squeeze(0) if single_input else aligned
