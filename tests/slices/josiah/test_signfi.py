"""Tests for src.slices.josiah.signfi.

These tests cover only the pure data-shaping functions; the .mat-file loader
is exercised by the actual training runs (test environment doesn't have the
1.37 GB SignFi-Home .mat file).
"""

from __future__ import annotations

import numpy as np
import torch

from src.slices.josiah.signfi import _to_windows, k_shot_split


def test_to_windows_shape_and_order():
    x = torch.arange(2 * 3 * 30 * 200, dtype=torch.float32).reshape(2, 3, 30, 200)
    out = _to_windows(x, num_windows=20)
    assert out.shape == (2, 20, 3, 30, 10)
    # First window of sample 0 should be x[0, :, :, 0:10].
    assert torch.equal(out[0, 0], x[0, :, :, 0:10])
    # Last window of sample 0 should be x[0, :, :, 190:200].
    assert torch.equal(out[0, 19], x[0, :, :, 190:200])


def test_to_windows_rejects_indivisible_nt():
    x = torch.randn(1, 3, 30, 197)
    try:
        _to_windows(x, num_windows=20)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible Nt")


def test_k_shot_split_balanced():
    # 5 classes, 10 samples each, k=3 -> 15 train, 35 test.
    labels = np.repeat(np.arange(5), 10)
    train_idx, test_idx = k_shot_split(labels, k=3, num_classes=5, seed=0)
    assert train_idx.size == 15
    assert test_idx.size == 35
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
    for c in range(5):
        train_c = labels[train_idx] == c
        assert train_c.sum() == 3


def test_k_shot_split_caps_at_class_size():
    """If k exceeds samples-per-class, take everything; no test samples for that class."""
    labels = np.array([0, 0, 1, 1, 1])  # class 0 has 2, class 1 has 3.
    train_idx, test_idx = k_shot_split(labels, k=4, num_classes=2, seed=0)
    assert train_idx.size == 5  # all samples used for training (k>=class size)
    assert test_idx.size == 0


def test_k_shot_split_deterministic_seed():
    labels = np.repeat(np.arange(10), 10)
    t1, _ = k_shot_split(labels, k=3, num_classes=10, seed=42)
    t2, _ = k_shot_split(labels, k=3, num_classes=10, seed=42)
    assert np.array_equal(t1, t2)
