import torch


def sample_categorical(categorical_probs, temperature=1.0):
    """
    Sample from a categorical distribution with temperature scaling.

    Args:
        categorical_probs: Unnormalized probabilities
        temperature: Temperature parameter for scaling logits. Higher values produce more uniform samples.

    Returns:
        samples: Sampled indices from the categorical distribution
        normalized_probs: The normalized probabilities used for sampling
    """
    original_shape = categorical_probs.size()[:-1]
    num_categories = categorical_probs.size(-1)
    flat_probs = categorical_probs.reshape(-1, num_categories)

    # Apply temperature scaling and normalize
    if temperature != 1.0:
        flat_probs = flat_probs.pow(1.0 / temperature)
    normalized_probs = torch.nn.functional.normalize(flat_probs, p=1, dim=-1)

    # Sample from the normalized distribution
    samples = torch.multinomial(normalized_probs, num_samples=1).squeeze(-1)

    return samples.view(original_shape), normalized_probs.view(*original_shape, num_categories)


def sample_edges(
    E: torch.Tensor, p_e0: torch.Tensor, lengths: torch.Tensor, argmax: bool = False
) -> torch.Tensor:
    """
    Build a one‑hot edge tensor that keeps any already‑denoised incoming edge
    and, for nodes still missing one, samples exactly ONE (parent, type) pair
    from the score logits.
    Args
    ----
    E       : (B, n, n, R)  current (partly‑denoised) one‑hot edges
    p_e0    : (B, n, n, R)  score logits; last‑2 channels = [no‑edge, masked]
    lengths : (B,)          number of real nodes per graph (padding after that)
    argmax  : bool          if True, take argmax instead of sampling (default False)
    Returns
    -------
    E_out   : (B, n, n, R)  one‑hot, upper/lower triangles mirrored, obeying
            • every j>0 has exactly one incoming edge i<j (or is padded)
            • channel R‑2 holds "no‑edge", channel R‑1 left for "masked"
    """

    # 1: Initialize output tensor and extract dimensions
    B, n, _, R = p_e0.shape
    dev = p_e0.device
    real_R = R - 2
    E_out = torch.zeros_like(p_e0)

    # 2: Identify existing edges and extract scores
    existing_mask = E[..., :real_R].sum(-1) > 0
    scores = p_e0[..., :real_R]

    # 3: Process each node to ensure exactly one incoming edge
    for j in range(1, n):
        has_edge = existing_mask[:, :j, j].any(1)
        need_edge = ~has_edge

        # 3.1: Copy existing edges
        if has_edge.any():
            rows_h = torch.arange(B, device=dev)[has_edge]
            parent_h = existing_mask[has_edge, :j, j].float().argmax(1)
            edge_h = (E[has_edge, :j, j, :real_R].float().argmax(-1))[
                torch.arange(parent_h.size(0), device=dev), parent_h
            ]
            E_out[rows_h, parent_h, j, edge_h] = 1

        # 3.2: Sample new edges where needed
        if need_edge.any():
            rows_n = torch.arange(B, device=dev)[need_edge]
            cand = scores[need_edge, :j, j, :]
            flat = cand.reshape(rows_n.size(0), -1)
            zero_row = flat.sum(1) == 0
            flat[zero_row] = 1.0
            if argmax:
                idx = flat.argmax(dim=1)
            else:
                idx = torch.multinomial(flat, 1).squeeze(1)
            k_new = idx.remainder(real_R)
            i_new = idx // real_R
            E_out[rows_n, i_new, j, k_new] = 1

    # 4: Mark untouched positions as no-edge
    none_mask = E_out.sum(-1, keepdim=True) == 0
    E_out[..., -2] = none_mask.squeeze(-1).float()

    # 5: Zero-out padding beyond graph length
    if lengths is not None:
        for b in range(B):
            l = lengths[b].item()
            if l < n:
                E_out[b, l:, :, :] = 0
                E_out[b, :, l:, :] = 0
                E_out[b, l:, :, -2] = 1
                E_out[b, :, l:, -2] = 1

    # 6: Mirror upper triangle for symmetry
    upper = torch.triu(E_out.permute(0, 3, 1, 2))
    upper = upper.bool()
    full = upper | upper.transpose(-1, -2)
    E_out = full.permute(0, 2, 3, 1).float()

    return E_out
