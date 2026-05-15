"""Augmentations for Slice 4 (coherence-aware subcarrier masking).

T4.5 ships `coherent_block_mask` — the slice's signature augmentation:
zero out a contiguous block of `block_width` subcarriers at a random
start position. `block_width` is the coherence-bandwidth estimate from
`coherence.py`; the block simulates a realistic frequency-selective
fading event where a coherent chunk of the spectrum drops out together.

The hand-crafted-aug baseline functions (`gaussian_noise`,
`random_subcarrier_mask`, `gaussian_then_mask`) are included so this
slice can run its `simclr-handcrafted` comparison mode self-contained,
matching Slice 1 / Slice 5 parameters exactly.
"""

from __future__ import annotations

import torch


def coherent_block_mask(
    x: torch.Tensor,
    block_width: int = 5,
) -> torch.Tensor:
    """Mask a contiguous block of `block_width` subcarriers.

    Args:
        x: `(T, S, A)` single sample or `(B, T, S, A)` batched.
        block_width: number of contiguous subcarriers to zero. Set to
            the coherence-bandwidth estimate from `coherence.py`.

    Returns:
        Same shape as `x`. Per-sample random start; two consecutive
        calls produce two views differing in their mask locations.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")
    s = x.shape[-2]
    if block_width <= 0:
        return x
    if block_width >= s:
        # Masking everything makes the view degenerate; clamp.
        block_width = s - 1
    max_start = s - block_width

    if x.ndim == 3:
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        out = x.clone()
        out[:, start : start + block_width, :] = 0
        return out

    b = x.shape[0]
    starts = torch.randint(0, max_start + 1, (b,), device=x.device)
    out = x.clone()
    for i in range(b):
        s_i = int(starts[i].item())
        out[i, :, s_i : s_i + block_width, :] = 0
    return out


# ---------------------------------------------------------------------------
# Hand-crafted-aug baseline (same as Slice 1 / Slice 5)


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def random_subcarrier_mask(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    if x.ndim == 3:
        s = x.shape[1]
        keep = torch.rand(s, device=x.device) >= p
        return x * keep.view(1, s, 1).to(x.dtype)
    if x.ndim == 4:
        b, _, s, _ = x.shape
        keep = torch.rand(b, s, device=x.device) >= p
        return x * keep.view(b, 1, s, 1).to(x.dtype)
    raise ValueError(
        f"random_subcarrier_mask expects (T, S, A) or (B, T, S, A); "
        f"got {tuple(x.shape)}"
    )


def gaussian_then_mask(
    x: torch.Tensor, sigma: float = 0.05, p: float = 0.15
) -> torch.Tensor:
    return random_subcarrier_mask(gaussian_noise(x, sigma=sigma), p=p)
