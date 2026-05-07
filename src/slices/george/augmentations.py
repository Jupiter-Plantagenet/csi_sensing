"""CSI augmentations for SimCLR view-pair generation.

T1.4 ships `doppler_warp` — a physics-informed time-axis stretch by a random
factor in [0.7, 1.4]. The intuition: activity speed scales the Doppler
component of CSI approximately linearly, so warping the time axis simulates
the same activity performed at a different speed and forces the encoder
to learn speed-invariant features.

The function operates on float-real CSI shaped (T, S, A) for a single sample
or (B, T, S, A) for a batch. T1.2 will decide how to project complex
Widar3.0 CSI into a real tensor (magnitude / real+imag stack); the
augmentation accepts whatever real-valued shape lands.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_FACTOR_RANGE = (0.7, 1.4)


def _sample_factor(low: float = 0.7, high: float = 1.4) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _doppler_warp_one(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Stretch a single CSI sample's time axis by `factor`, then crop or
    zero-pad back to the original length.

    Cropping (rather than re-resampling to T) preserves the *frequency*
    content of the warped signal: a factor>1 stretch shifts the dominant
    frequency *down* in the cropped result; a factor<1 squeeze shifts it
    *up*. T1.5's sanity test exercises this property.
    """
    if x.ndim != 3:
        raise ValueError(f"_doppler_warp_one expects (T, S, A); got {tuple(x.shape)}")
    t, s, a = x.shape
    t_new = max(1, int(round(t * factor)))

    # F.interpolate with mode='linear' wants (N, C, L). Treat S*A as channels,
    # T as the spatial axis.
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
    """Apply Doppler-aware time warping to a CSI tensor.

    Accepts a single sample shaped `(T, S, A)` or a batch shaped
    `(B, T, S, A)`. If `factor` is None, each sample gets an independently
    sampled factor from `factor_range` — that's what makes this useful as a
    SimCLR view-pair augmentation: two calls produce two views that differ
    in their Doppler stretch.
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
