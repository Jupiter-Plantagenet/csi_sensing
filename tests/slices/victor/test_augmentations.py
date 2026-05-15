"""Tests for Slice 6 in-slice augmentation reimplementations (T6.3).

These mirror the spirit of the Slice 1 (Doppler) and Slice 4
(coherent-mask) tests but exercise this slice's local copies — the
point of T6.3 is that the reimplementations are independent.
"""

from __future__ import annotations

import pytest
import torch

from src.slices.victor.augmentations import (
    coherent_block_mask,
    doppler_then_coherent_mask,
    doppler_warp,
)
from src.slices.victor.coherence import estimate_coherence_bandwidth_subcarriers


# --- Doppler -------------------------------------------------------------


def test_doppler_warp_preserves_shape_single() -> None:
    x = torch.randn(10, 30, 3)
    out = doppler_warp(x, factor=0.8)
    assert out.shape == x.shape


def test_doppler_warp_preserves_shape_batch() -> None:
    x = torch.randn(4, 10, 30, 3)
    out = doppler_warp(x, factor=1.2)
    assert out.shape == x.shape


def test_doppler_warp_two_calls_differ() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 10, 30, 3)
    v1 = doppler_warp(x)
    v2 = doppler_warp(x)
    assert not torch.equal(v1, v2)


def test_doppler_warp_rejects_garbage_shape() -> None:
    with pytest.raises(ValueError):
        doppler_warp(torch.zeros(30, 3))


# --- Coherence bandwidth + coherent mask ---------------------------------


def test_coherence_independent_subcarriers() -> None:
    torch.manual_seed(0)
    x = torch.randn(50, 30, 3)
    bw = estimate_coherence_bandwidth_subcarriers(x, threshold=0.5)
    assert bw == 1


def test_coherent_block_mask_zeros_contiguous() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = coherent_block_mask(x, block_width=5)
    assert out.shape == x.shape
    for i in range(2):
        zero_mask = (out[i].amax(dim=(0, 2)) == 0).int()
        assert int(zero_mask.sum()) == 5
        idx = torch.where(zero_mask == 1)[0]
        assert int(idx[-1] - idx[0]) == 4


def test_coherent_block_mask_passthrough_when_zero_width() -> None:
    x = torch.randn(2, 10, 30, 3)
    out = coherent_block_mask(x, block_width=0)
    assert torch.equal(out, x)


# --- Composition ---------------------------------------------------------


def test_doppler_then_coherent_mask_shape() -> None:
    x = torch.randn(4, 10, 30, 3)
    out = doppler_then_coherent_mask(x, block_width=5)
    assert out.shape == x.shape


def test_doppler_then_coherent_mask_actually_masks() -> None:
    """Composition should still zero out a contiguous subcarrier block."""
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = doppler_then_coherent_mask(x, block_width=5)
    # Each sample must have at least 5 fully-zero subcarriers (Doppler warp
    # on a constant signal stays constant within the support region, so the
    # masked subcarriers stay exactly zero).
    for i in range(2):
        zero_count = int((out[i].amax(dim=(0, 2)) == 0).sum())
        assert zero_count >= 5, f"sample {i}: expected >=5 zeros, got {zero_count}"
