"""Augmentations for Slice 6 (composability of Doppler + coherence-mask).

T6.3 reimplements both physics-informed augmentations from scratch
in-slice so this slice is independent of Slices 1 and 4:

- `doppler_warp`: time-axis stretch by a random factor in [0.7, 1.4].
  Activity speed scales the Doppler component of CSI approximately
  linearly; warping time simulates the same activity at a different
  speed.
- `coherent_block_mask`: zero a contiguous block of `block_width`
  subcarriers at a random start. `block_width` is the coherence-bw
  estimate from `coherence.py` — the block simulates a frequency-
  selective fade where a coherent chunk of the spectrum drops out.

Composition helpers (`doppler_then_coherent_mask` and
`doppler_or_coherent_mask`) are the candidates the composability study
(T6.5) uses to combine the two.

Generic hand-crafted baseline (`gaussian_then_mask`) is included for the
`simclr-handcrafted` comparison column.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_FACTOR_RANGE = (0.7, 1.4)


# ---------------------------------------------------------------------------
# Doppler-aware time warping


def _sample_factor(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _doppler_warp_one(x: torch.Tensor, factor: float) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"_doppler_warp_one expects (T, S, A); got {tuple(x.shape)}")
    t, s, a = x.shape
    t_new = max(1, int(round(t * factor)))
    x_in = x.permute(1, 2, 0).reshape(1, s * a, t)
    x_r = F.interpolate(x_in, size=t_new, mode="linear", align_corners=False)
    x_r = x_r.reshape(s, a, t_new).permute(2, 0, 1)
    if x_r.shape[0] >= t:
        return x_r[:t]
    pad = torch.zeros(t - x_r.shape[0], s, a, dtype=x.dtype, device=x.device)
    return torch.cat([x_r, pad], dim=0)


def _doppler_warp_batched(
    x: torch.Tensor, factors: torch.Tensor
) -> torch.Tensor:
    """Vectorized Doppler-warp over a batch using ``F.grid_sample``.

    Equivalent to applying ``_doppler_warp_one`` per sample with
    ``factor=factors[i]``, but without the Python per-sample loop. Reads
    each output time index ``k`` from source index ``k / factors[i]``;
    positions beyond ``T-1`` are zero (matches the pad-after-crop
    behaviour of the loop variant).
    """
    b, t, s, a = x.shape
    sa = s * a
    src = torch.arange(t, device=x.device, dtype=x.dtype)
    src_idx = src.unsqueeze(0) / factors.unsqueeze(1)  # (B, T)
    valid = (src_idx <= (t - 1)).to(x.dtype)  # (B, T)
    src_clamped = src_idx.clamp(min=0.0, max=float(t - 1))
    grid_y = 2.0 * src_clamped / max(1, t - 1) - 1.0  # (B, T)
    grid_y = grid_y.unsqueeze(2).expand(b, t, sa)
    if sa > 1:
        col = torch.arange(sa, device=x.device, dtype=x.dtype)
        grid_x = 2.0 * col / (sa - 1) - 1.0
    else:
        grid_x = torch.zeros(sa, device=x.device, dtype=x.dtype)
    grid_x = grid_x.view(1, 1, sa).expand(b, t, sa)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, T, SA, 2)
    img = x.permute(0, 1, 2, 3).reshape(b, t, s, a).reshape(b, t, sa).unsqueeze(1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.squeeze(1).reshape(b, t, s, a)
    return out * valid.view(b, t, 1, 1)


def doppler_warp(
    x: torch.Tensor,
    factor: float | None = None,
    factor_range: tuple[float, float] = DEFAULT_FACTOR_RANGE,
) -> torch.Tensor:
    """Doppler-aware time warp.

    Accepts `(T, S, A)` or `(B, T, S, A)`. If `factor` is None, each
    sample gets an independent factor from `factor_range`. The batched
    path is fully vectorized via ``F.grid_sample`` (no Python per-sample
    loop) so the augmentation stays on-device on GPU.
    """
    low, high = factor_range
    if x.ndim == 3:
        f = _sample_factor(low, high) if factor is None else factor
        return _doppler_warp_one(x, f)
    if x.ndim == 4:
        b = x.shape[0]
        if factor is None:
            factors = torch.empty(b, device=x.device, dtype=x.dtype).uniform_(low, high)
        else:
            factors = torch.full((b,), float(factor), device=x.device, dtype=x.dtype)
        return _doppler_warp_batched(x, factors)
    raise ValueError(
        f"doppler_warp expects (T, S, A) or (B, T, S, A); got {tuple(x.shape)}"
    )


# ---------------------------------------------------------------------------
# Coherence-aware subcarrier masking


def coherent_block_mask(
    x: torch.Tensor,
    block_width: int = 5,
) -> torch.Tensor:
    """Mask a contiguous block of `block_width` subcarriers at random start.

    Accepts `(T, S, A)` or `(B, T, S, A)`.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")
    s = x.shape[-2]
    if block_width <= 0:
        return x
    if block_width >= s:
        block_width = s - 1
    max_start = s - block_width

    if x.ndim == 3:
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        out = x.clone()
        out[:, start : start + block_width, :] = 0
        return out

    # Vectorized: build a per-sample (b, s) boolean mask.
    b = x.shape[0]
    starts = torch.randint(0, max_start + 1, (b,), device=x.device)
    s_idx = torch.arange(s, device=x.device)
    in_block = (s_idx.unsqueeze(0) >= starts.unsqueeze(1)) & (
        s_idx.unsqueeze(0) < (starts + block_width).unsqueeze(1)
    )  # (b, s)
    keep = (~in_block).to(x.dtype)
    return x * keep.view(b, 1, s, 1)


# ---------------------------------------------------------------------------
# Composition strategies for T6.5


def doppler_then_coherent_mask(
    x: torch.Tensor,
    block_width: int = 5,
    factor_range: tuple[float, float] = DEFAULT_FACTOR_RANGE,
) -> torch.Tensor:
    """Sequential composition: Doppler warp, then coherent block mask.

    Both transformations applied to the same view. This is the "do they
    compose" candidate the composability study compares against the
    individual augmentations.
    """
    return coherent_block_mask(
        doppler_warp(x, factor_range=factor_range),
        block_width=block_width,
    )


# ---------------------------------------------------------------------------
# Hand-crafted baseline (same as Slices 1 / 4 / 5)


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
