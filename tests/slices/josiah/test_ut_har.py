"""Tests for src.slices.josiah.ut_har and the AutoFi UT-HAR adapter."""

from __future__ import annotations

import numpy as np
import torch

from src.slices.josiah.ut_har import (
    NUM_UT_HAR_CLASSES,
    UT_HAR_F,
    UT_HAR_T,
    ut_har_k_shot_indices,
)


def test_k_shot_indices_balanced():
    labels = np.repeat(np.arange(NUM_UT_HAR_CLASSES), 50)  # 7 cls × 50
    idx = ut_har_k_shot_indices(labels, k=20, num_classes=NUM_UT_HAR_CLASSES, seed=42)
    assert idx.size == 7 * 20
    for c in range(NUM_UT_HAR_CLASSES):
        assert (labels[idx] == c).sum() == 20


def test_k_shot_indices_caps_at_class_size():
    labels = np.array([0, 0, 1, 1, 1])  # cls 0: 2 samples, cls 1: 3 samples
    idx = ut_har_k_shot_indices(labels, k=10, num_classes=2, seed=42)
    # k>class_size -> take everything available per class.
    assert idx.size == 5


def test_k_shot_indices_deterministic_seed():
    labels = np.repeat(np.arange(7), 100)
    a = ut_har_k_shot_indices(labels, k=20, num_classes=7, seed=42)
    b = ut_har_k_shot_indices(labels, k=20, num_classes=7, seed=42)
    assert np.array_equal(a, b)


def test_autofi_uthar_encoder_supervised_shape():
    from src.slices.josiah.autofi import AutoFiUTHAREncoder

    enc = AutoFiUTHAREncoder(hidden_states=256)
    x = torch.randn(4, 1, UT_HAR_T, UT_HAR_F)
    out = enc(x, flag="supervised")
    assert out.shape == (4, enc.feat_dim)


def test_autofi_uthar_encoder_unsupervised_shape():
    from src.slices.josiah.autofi import AutoFiUTHAREncoder

    enc = AutoFiUTHAREncoder(hidden_states=256)
    x = torch.randn(4, 1, UT_HAR_T, UT_HAR_F)
    out = enc(x)
    assert out.shape == (4, 256)


def test_autofi_uthar_parallel_supervised_returns_logits():
    from src.slices.josiah.autofi import AutoFiUTHARParallel

    model = AutoFiUTHARParallel(num_classes=NUM_UT_HAR_CLASSES, hidden_states=256)
    x = torch.randn(4, 1, UT_HAR_T, UT_HAR_F)
    y1, y2 = model(x, x, flag="supervised")
    assert y1.shape == (4, NUM_UT_HAR_CLASSES)
    assert y2.shape == (4, NUM_UT_HAR_CLASSES)


def test_pretrain_autofi_uthar_runs_one_epoch():
    from torch.utils.data import DataLoader, TensorDataset

    from src.slices.josiah.autofi import AutoFiUTHARParallel, pretrain_autofi_uthar

    torch.manual_seed(0)
    x = torch.randn(8, 1, UT_HAR_T, UT_HAR_F)
    y = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = AutoFiUTHARParallel(num_classes=NUM_UT_HAR_CLASSES, hidden_states=64)
    history = pretrain_autofi_uthar(model, loader, epochs=1, device="cpu", log_every=0)
    assert len(history) == 1
    assert history[0] == history[0]  # finite
