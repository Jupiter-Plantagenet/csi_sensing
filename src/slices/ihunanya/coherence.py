"""Coherence-bandwidth estimation for Slice 4 (T4.3).

The slice masks blocks of contiguous subcarriers matching the channel's
coherence bandwidth — the frequency width over which adjacent subcarriers
remain correlated (they fade together rather than independently). The
block width is the integer "how many subcarriers stay coherent" rather
than a continuous Hz value, since the encoder operates on a fixed
30-subcarrier discrete grid.

Two estimators, both supported:

1. **Frequency-domain autocorrelation** (default). For each sample,
   compute the lagged subcarrier-vs-subcarrier autocorrelation of the
   magnitude. The smallest lag at which |autocorr| drops below a
   threshold (default 0.5) is the coherence bandwidth in subcarriers.
   Works on the real-valued magnitude-CSI we already have.

2. **CIR-based delay spread** (alternative). Take the inverse FFT along
   the subcarrier axis to get the channel impulse response; compute the
   RMS delay spread; coherence bandwidth ≈ 1 / (5 · τ_RMS). Requires
   complex CSI — not directly applicable to our magnitude projection,
   so this estimator is provided for future use if the loader is
   extended to keep complex CSI.

For our default setup (30 subcarriers, magnitude-only CSI on Widar3.0),
the autocorrelation estimator is the operative one.
"""

from __future__ import annotations

import torch


def estimate_coherence_bandwidth_subcarriers(
    x: torch.Tensor,
    threshold: float = 0.5,
    min_width: int = 1,
    max_width: int | None = None,
) -> int:
    """Estimate coherence bandwidth in *number of subcarriers*.

    Args:
        x: real-valued CSI shaped `(T, S, A)` or `(B, T, S, A)`. The
            estimator averages over time and antennas (and batch, if
            present) before computing the autocorrelation, so the
            returned scalar is a dataset-wide estimate.
        threshold: lag-correlation threshold defining "still coherent."
            Default 0.5 matches the convention used in coherence-bandwidth
            literature; some texts use 0.9 for tighter coherence.
        min_width: minimum returned width. Defaults to 1 (no smaller
            block makes sense for masking).
        max_width: optional cap. Defaults to `S // 2` — beyond that,
            "coherent block" loses meaning.

    Returns:
        Integer coherence-bandwidth width in subcarriers.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)  # (1, T, S, A)
    if x.ndim != 4:
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")
    b, t, s, a = x.shape
    if max_width is None:
        max_width = s // 2

    # Flatten batch/time/antennas; treat each as one realization.
    # Demean over the subcarrier axis to compute correlation, not covariance.
    flat = x.permute(0, 1, 3, 2).reshape(b * t * a, s)
    flat = flat - flat.mean(dim=1, keepdim=True)

    # Variance per realization
    raw_var = (flat * flat).mean(dim=1)
    var = raw_var + 1e-12

    # Edge case: if the input is essentially constant along the subcarrier
    # axis (degenerate "perfectly coherent" case), every lag stays at full
    # correlation by definition. Return max_width directly.
    if raw_var.mean().item() < 1e-10:
        return max(min_width, max_width)

    # Autocorrelation at each lag in [0, max_width].
    auto = []
    for lag in range(0, max_width + 1):
        if lag == 0:
            corr = torch.ones(flat.shape[0], dtype=flat.dtype, device=flat.device)
        else:
            corr = (flat[:, :-lag] * flat[:, lag:]).mean(dim=1) / var
        auto.append(corr.mean().item())

    # Find the smallest lag where autocorrelation drops below threshold.
    for lag, value in enumerate(auto):
        if lag == 0:
            continue
        if value < threshold:
            return max(min_width, lag)
    return max(min_width, max_width)


def autocorrelation_curve(
    x: torch.Tensor,
    max_lag: int | None = None,
) -> torch.Tensor:
    """Return the average lagged autocorrelation as a 1-D tensor of length `max_lag + 1`.

    Useful for diagnostic plots in T4.6 and the per-slice writeup. lag=0
    is always 1.0 by construction.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")
    b, t, s, a = x.shape
    if max_lag is None:
        max_lag = s - 1
    flat = x.permute(0, 1, 3, 2).reshape(b * t * a, s)
    flat = flat - flat.mean(dim=1, keepdim=True)
    var = (flat * flat).mean(dim=1) + 1e-12

    curve = torch.zeros(max_lag + 1, dtype=flat.dtype)
    curve[0] = 1.0
    for lag in range(1, max_lag + 1):
        corr = (flat[:, :-lag] * flat[:, lag:]).mean(dim=1) / var
        curve[lag] = corr.mean().item()
    return curve
