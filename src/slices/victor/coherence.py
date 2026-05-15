"""Coherence-bandwidth estimation for Slice 6 (T6.3 prerequisite).

In-slice reimplementation of the coherence-bandwidth estimator. The
composability slice needs its own copy so it's independent of Slice 4 —
no cross-slice imports per repo convention.

Estimator: lagged subcarrier autocorrelation. For each sample, compute
the correlation between subcarrier `k` and subcarrier `k+lag` along the
subcarrier axis, averaged over time / antennas / batch. The smallest
lag at which the average autocorrelation falls below `threshold`
(default 0.5) is the coherence bandwidth, expressed in number of
subcarriers — the integer width of a "still coherent" block on our
30-subcarrier grid.
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
        x: real-valued CSI shaped `(T, S, A)` or `(B, T, S, A)`.
        threshold: lag-correlation threshold defining "still coherent."
        min_width: minimum returned width (default 1).
        max_width: optional cap (default `S // 2`).

    Returns:
        Integer coherence-bandwidth width in subcarriers.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected (T, S, A) or (B, T, S, A); got ndim={x.ndim}")
    b, t, s, a = x.shape
    if max_width is None:
        max_width = s // 2

    flat = x.permute(0, 1, 3, 2).reshape(b * t * a, s)
    flat = flat - flat.mean(dim=1, keepdim=True)

    raw_var = (flat * flat).mean(dim=1)
    var = raw_var + 1e-12

    # Degenerate "perfectly coherent" input (constant along subcarrier axis):
    # every lag stays at full correlation by definition. Return max_width.
    if raw_var.mean().item() < 1e-10:
        return max(min_width, max_width)

    for lag in range(1, max_width + 1):
        corr = (flat[:, :-lag] * flat[:, lag:]).mean(dim=1) / var
        if corr.mean().item() < threshold:
            return max(min_width, lag)
    return max(min_width, max_width)
