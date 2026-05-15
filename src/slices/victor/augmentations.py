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


def doppler_warp(
    x: torch.Tensor,
    factor: float | None = None,
    factor_range: tuple[float, float] = DEFAULT_FACTOR_RANGE,
) -> torch.Tensor:
    """Doppler-aware time warp.

    Accepts `(T, S, A)` or `(B, T, S, A)`. If `factor` is None, each
    sample gets an independent factor from `factor_range`.
    """
    low, high = factor_range
    if x.ndim == 3:
        f = _sample_factor(low, high) if factor is None else factor
        return _doppler_warp_one(x, f)
    if x.ndim == 4:
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            f = _sample_factor(low, high) if factor is None else factor
            out[i] = _doppler_warp_one(x[i], f)
        return out
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

    b = x.shape[0]
    starts = torch.randint(0, max_start + 1, (b,), device=x.device)
    out = x.clone()
    for i in range(b):
        s_i = int(starts[i].item())
        out[i, :, s_i : s_i + block_width, :] = 0
    return out


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
