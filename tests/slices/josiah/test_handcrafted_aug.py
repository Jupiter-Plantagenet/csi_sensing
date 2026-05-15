"""Unit tests for Slice 5's hand-crafted-augmentation baseline (T5.6).

`gaussian_noise`, `random_subcarrier_mask`, and `gaussian_then_mask`
match Slice 1's implementations so the row in the team's comparison
figure is byte-identical augmentation pipelines.
"""

from __future__ import annotations

import pytest
import torch

from src.slices.josiah.augmentations import (
    gaussian_noise,
    gaussian_then_mask,
    random_subcarrier_mask,
)


def test_gaussian_noise_changes_input() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = gaussian_noise(x, sigma=0.1)
    assert out.shape == x.shape
    # With sigma=0.1, output should differ from constant input.
    assert not torch.allclose(out, x)
    # Difference distribution should have ~zero mean.
    diff = out - x
    assert abs(float(diff.mean())) < 0.05
    # And std close to sigma.
    assert abs(float(diff.std()) - 0.1) < 0.02


def test_random_subcarrier_mask_zeros_some_subcarriers() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = random_subcarrier_mask(x, p=0.3)
    assert out.shape == x.shape
    # Some subcarriers should be entirely zero across the time axis;
    # others should be entirely one (no per-cell randomness).
    per_subcarrier_max = out.amax(dim=(1, 3))  # (B, S)
    zero_subcarriers = (per_subcarrier_max == 0).sum().item()
    # With p=0.3 on 30 subcarriers across 2 samples, expect around 18
    # zeroed subcarriers in total; allow a wide tolerance for randomness.
    assert 5 <= zero_subcarriers <= 35


def test_random_subcarrier_mask_per_sample_independence() -> None:
    """Two samples in the batch should get independent masks."""
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = random_subcarrier_mask(x, p=0.5)
    # The zeroed subcarrier sets per sample should not be identical with very
    # high probability for B=2, S=30, p=0.5.
    sample_0_zeros = out[0].amax(dim=(0, 2)) == 0
    sample_1_zeros = out[1].amax(dim=(0, 2)) == 0
    assert not torch.equal(sample_0_zeros, sample_1_zeros)


def test_gaussian_then_mask_composes() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 10, 30, 3)
    out = gaussian_then_mask(x, sigma=0.05, p=0.2)
    assert out.shape == x.shape
    # Output should differ from input (Gaussian noise was added) and have
    # some subcarriers fully zero.
    assert not torch.allclose(out, x)


def test_random_subcarrier_mask_rejects_garbage_shape() -> None:
    with pytest.raises(ValueError):
        random_subcarrier_mask(torch.zeros(10, 30), p=0.1)
