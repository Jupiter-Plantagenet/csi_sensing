"""Tests for src.slices.collins.bvp_velocity_jitter."""

from __future__ import annotations

import torch

from src.slices.collins.bvp_velocity_jitter import bvp_velocity_jitter


def test_jitter_preserves_shape_single():
    x = torch.randn(22, 20, 20)
    y = bvp_velocity_jitter(x)
    assert y.shape == x.shape


def test_jitter_preserves_shape_batch():
    x = torch.randn(4, 22, 20, 20)
    y = bvp_velocity_jitter(x)
    assert y.shape == x.shape


def test_jitter_identity_when_zero_params():
    x = torch.randn(22, 20, 20)
    y = bvp_velocity_jitter(x, angle_range_deg=(0.0, 0.0), shift_range_cells=(0.0, 0.0))
    # At zero rotation and zero shift the sampling grid is the identity map;
    # the result equals the input up to bilinear-sample boundary effects.
    assert torch.allclose(y, x, atol=1e-4)


def test_jitter_two_views_differ():
    """Two independent calls must produce different views (so SimCLR has signal)."""
    torch.manual_seed(0)
    x = torch.randn(4, 22, 20, 20)
    a = bvp_velocity_jitter(x)
    b = bvp_velocity_jitter(x)
    assert not torch.allclose(a, b)


def test_jitter_rejects_garbage_shape():
    with __import__("pytest").raises(ValueError):
        bvp_velocity_jitter(torch.randn(20, 20))
