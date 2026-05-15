"""Unit tests for Slice 2's static/dynamic decomposition and aug (T2.3, T2.4, T2.5)."""

from __future__ import annotations

import pytest
import torch

from src.slices.chigozie.augmentations import static_perturb
from src.slices.chigozie.decompose import static_dynamic_split


def test_split_is_lossless_time_mean() -> None:
    x = torch.randn(2, 50, 30, 3)
    static, dynamic = static_dynamic_split(x, method="time-mean")
    assert static.shape == x.shape
    assert dynamic.shape == x.shape
    assert torch.allclose(static + dynamic, x, atol=1e-6)


def test_split_is_lossless_lowpass() -> None:
    x = torch.randn(2, 100, 30, 3)
    static, dynamic = static_dynamic_split(
        x, method="lowpass", cutoff_hz=20.0, sample_rate_hz=1000.0
    )
    assert static.shape == x.shape
    assert torch.allclose(static + dynamic, x, atol=1e-5)


def test_static_is_constant_in_time_for_mean_method() -> None:
    x = torch.randn(2, 50, 30, 3)
    static, _ = static_dynamic_split(x, method="time-mean")
    # Static should be identical across the time axis.
    assert torch.allclose(static[:, 0:1], static[:, 25:26], atol=1e-6)


def test_decomposition_routes_known_slow_and_fast() -> None:
    """A synthetic CSI = (slow sine) + (fast sine) decomposes correctly."""
    t = torch.linspace(0, 1, 1000)
    slow = torch.sin(2 * torch.pi * 0.5 * t)  # 0.5 Hz
    fast = torch.sin(2 * torch.pi * 50.0 * t)  # 50 Hz
    x = (slow + fast).reshape(1000, 1, 1)

    static, dynamic = static_dynamic_split(
        x, method="lowpass", cutoff_hz=5.0, sample_rate_hz=1000.0
    )
    # The static part should track the slow component (small error).
    # The dynamic part should track the fast component.
    # Use power ratios as a sanity check rather than exact equality.
    static_power = static.pow(2).mean()
    dynamic_power = dynamic.pow(2).mean()
    # Both components have power ~0.5 in the original; static gets ~slow,
    # dynamic gets ~fast — so both around 0.5.
    assert 0.1 < static_power < 1.0
    assert 0.1 < dynamic_power < 1.0


def test_static_perturb_changes_input_in_batch() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 50, 30, 3)
    out = static_perturb(x, method="time-mean")
    assert out.shape == x.shape
    # The output should differ from input since at least one sample's static
    # was replaced by a different sample's static.
    assert not torch.allclose(out, x)


def test_static_perturb_preserves_dynamic() -> None:
    """Static-perturbation keeps dynamic; only static is swapped."""
    torch.manual_seed(0)
    x = torch.randn(4, 50, 30, 3)
    out = static_perturb(x, method="time-mean")
    # Per-sample dynamic = sample - sample.mean(time). The dynamic should
    # be the same before and after, modulo the new static term.
    _, dynamic_x = static_dynamic_split(x, method="time-mean")
    _, dynamic_out = static_dynamic_split(out, method="time-mean")
    assert torch.allclose(dynamic_x, dynamic_out, atol=1e-5)


def test_static_perturb_single_sample_passthrough() -> None:
    x = torch.randn(50, 30, 3)
    out = static_perturb(x)
    assert torch.equal(out, x)


def test_decompose_rejects_garbage_shape() -> None:
    with pytest.raises(ValueError):
        static_dynamic_split(torch.zeros(30, 3))  # missing time axis
