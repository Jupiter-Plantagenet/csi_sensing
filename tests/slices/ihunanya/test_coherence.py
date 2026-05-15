"""Unit tests for Slice 4's coherence-bandwidth and coherent-block-mask (T4.3, T4.4, T4.5).

T4.4 is the sanity test: a synthetic two-tap channel with a known delay
spread should produce a coherence-bandwidth estimate that lands in the
expected band.
"""

from __future__ import annotations

import pytest
import torch

from src.slices.ihunanya.augmentations import coherent_block_mask
from src.slices.ihunanya.coherence import (
    autocorrelation_curve,
    estimate_coherence_bandwidth_subcarriers,
)


def test_perfectly_flat_subcarriers_give_max_coherence() -> None:
    """If all subcarriers are identical, coherence bandwidth = max width."""
    # x: (T, S, A) = (10, 30, 3). All subcarriers equal a single time signal.
    base = torch.randn(10, 1, 3)
    x = base.expand(10, 30, 3).clone()
    bw = estimate_coherence_bandwidth_subcarriers(x, threshold=0.5)
    # With perfect correlation, the threshold is never crossed; returns max_width.
    assert bw == 30 // 2


def test_independent_subcarriers_give_min_coherence() -> None:
    """If subcarriers are independent random noise, coherence drops immediately."""
    torch.manual_seed(0)
    x = torch.randn(50, 30, 3)
    bw = estimate_coherence_bandwidth_subcarriers(x, threshold=0.5)
    # Lag-1 autocorrelation of i.i.d. noise is ~0; threshold 0.5 fails at lag 1.
    assert bw == 1


def test_partial_correlation_gives_intermediate_coherence() -> None:
    """A signal with gradual correlation should land between 1 and max."""
    # Construct an AR(1) signal across the subcarrier axis: each subcarrier
    # is a noisy version of the previous. The autocorrelation decays
    # geometrically with the AR coefficient.
    torch.manual_seed(0)
    rho = 0.9
    s = 30
    x_s = torch.zeros(20, s, 3)
    x_s[:, 0, :] = torch.randn(20, 3)
    for k in range(1, s):
        x_s[:, k, :] = rho * x_s[:, k - 1, :] + (1 - rho) * torch.randn(20, 3)
    bw = estimate_coherence_bandwidth_subcarriers(x_s, threshold=0.5)
    # AR(1) with rho=0.9: autocorr drops below 0.5 around lag log(0.5)/log(0.9) ≈ 6.6
    assert 3 <= bw <= 10, f"expected ~6 subcarriers, got {bw}"


def test_autocorrelation_curve_starts_at_one() -> None:
    x = torch.randn(10, 30, 3)
    curve = autocorrelation_curve(x, max_lag=10)
    assert curve.shape == (11,)
    assert abs(curve[0].item() - 1.0) < 1e-5


def test_coherent_block_mask_zeros_contiguous_subcarriers() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = coherent_block_mask(x, block_width=5)
    assert out.shape == x.shape
    # Per sample, exactly 5 contiguous subcarriers should be zero.
    for i in range(2):
        zero_mask = (out[i].amax(dim=(0, 2)) == 0).int()  # (S,)
        assert (
            int(zero_mask.sum()) == 5
        ), f"sample {i}: expected 5 zeros, got {int(zero_mask.sum())}"
        # And they should be contiguous.
        idx = torch.where(zero_mask == 1)[0]
        assert int(idx[-1] - idx[0]) == 4


def test_coherent_block_mask_two_calls_differ() -> None:
    """Two independent calls should mask different subcarrier ranges."""
    torch.manual_seed(0)
    x = torch.ones(4, 10, 30, 3)
    v1 = coherent_block_mask(x, block_width=5)
    v2 = coherent_block_mask(x, block_width=5)
    assert not torch.equal(v1, v2)


def test_coherent_block_mask_passthrough_when_zero_width() -> None:
    x = torch.randn(2, 10, 30, 3)
    out = coherent_block_mask(x, block_width=0)
    assert torch.equal(out, x)


def test_coherent_block_mask_rejects_garbage_shape() -> None:
    with pytest.raises(ValueError):
        coherent_block_mask(torch.zeros(30, 3), block_width=5)
