"""Static / dynamic decomposition for Slice 2.

T2.3 implements the decomposition the augmentation in T2.5 depends on.

A CSI sample over a short window can be split into a *static* component
(slow variation from walls, furniture, ceiling) and a *dynamic*
component (fast variation from the moving person):

    x(t, s, a) = static(t, s, a) + dynamic(t, s, a)

Two strategies, both supported:

1. **Time-mean** (default for short windows). `static(t) = mean_t(x)`,
   broadcast back across the time axis. `dynamic = x - static`. This is
   the limit of a temporal lowpass with cutoff → 0 Hz — appropriate when
   the time window is short relative to the body-motion timescale (here
   our default `T = 100` samples ≈ 100 ms is well below the gesture
   timescale).

2. **Lowpass** (for longer windows or whenever the room contribution
   has visible slow variation). A simple moving-average filter with a
   window proportional to the requested cutoff frequency. The doc-09
   default of 2 Hz corresponds to a window of `1000 // 2 = 500` samples
   at the 1000-packet/s Widar3.0 sample rate, which only makes sense for
   `T ≥ 500`. With our default `T = 100` this falls back to time-mean.

Both return `(static, dynamic)` tensors with the same shape as the input;
`dynamic = x - static` exactly, so the decomposition is lossless under
recombination.
"""

from __future__ import annotations

from typing import Literal

import torch

DecompMethod = Literal["time-mean", "lowpass"]
DEFAULT_LOWPASS_CUTOFF_HZ = 2.0
DEFAULT_SAMPLE_RATE_HZ = 1000.0


def static_dynamic_split(
    x: torch.Tensor,
    method: DecompMethod = "time-mean",
    cutoff_hz: float = DEFAULT_LOWPASS_CUTOFF_HZ,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split CSI into (static, dynamic) components.

    Args:
        x: `(T, S, A)` single sample or `(B, T, S, A)` batch.
        method: 'time-mean' (default) or 'lowpass'.
        cutoff_hz: lowpass cutoff frequency in Hz; only used when
            `method='lowpass'`.
        sample_rate_hz: CSI sample rate; only used for `lowpass`.

    Returns:
        `(static, dynamic)`, each the same shape as `x`. `static + dynamic`
        reconstructs `x` exactly.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")

    if method == "time-mean":
        static = _time_mean(x)
    elif method == "lowpass":
        static = _temporal_lowpass(x, cutoff_hz, sample_rate_hz)
    else:
        raise ValueError(f"unknown method: {method!r}")

    dynamic = x - static
    return static, dynamic


def _time_mean(x: torch.Tensor) -> torch.Tensor:
    """Time-mean broadcast back across the time axis."""
    if x.ndim == 3:
        # (T, S, A) -> (1, S, A) -> (T, S, A)
        m = x.mean(dim=0, keepdim=True)
        return m.expand_as(x).clone()
    # (B, T, S, A) -> (B, 1, S, A) -> (B, T, S, A)
    m = x.mean(dim=1, keepdim=True)
    return m.expand_as(x).clone()


def _temporal_lowpass(
    x: torch.Tensor, cutoff_hz: float, sample_rate_hz: float
) -> torch.Tensor:
    """Moving-average lowpass along the time axis.

    Window size: `sample_rate_hz / cutoff_hz`. If the window is wider than
    the available time axis, falls back to the time-mean.
    """
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive; got {cutoff_hz}")
    window = max(1, int(round(sample_rate_hz / cutoff_hz)))
    t = x.shape[-3]
    if window >= t:
        return _time_mean(x)

    # Convolve along the time axis with a uniform kernel. Treat (S, A) as
    # independent channels.
    if x.ndim == 3:
        _t, s, a = x.shape
        x_in = x.permute(1, 2, 0).reshape(1, s * a, t)  # (1, S*A, T)
        kernel = torch.ones(s * a, 1, window, dtype=x.dtype, device=x.device) / window
        # Pad symmetrically to keep length T.
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        x_pad = torch.nn.functional.pad(x_in, (pad_left, pad_right), mode="replicate")
        x_smooth = torch.nn.functional.conv1d(x_pad, kernel, groups=s * a)
        return x_smooth.reshape(s, a, t).permute(2, 0, 1)
    # (B, T, S, A)
    b, _t, s, a = x.shape
    x_in = x.permute(0, 2, 3, 1).reshape(b, s * a, t)
    kernel = torch.ones(s * a, 1, window, dtype=x.dtype, device=x.device) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = torch.nn.functional.pad(x_in, (pad_left, pad_right), mode="replicate")
    x_smooth = torch.nn.functional.conv1d(x_pad, kernel, groups=s * a)
    return x_smooth.reshape(b, s, a, t).permute(0, 3, 1, 2)
