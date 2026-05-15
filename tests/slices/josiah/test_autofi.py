"""Tests for src.slices.josiah.autofi."""

from __future__ import annotations

import torch

from src.slices.josiah.autofi import (
    AutoFiBVPEncoder,
    AutoFiGSSLoss,
    AutoFiParallel,
    gaussian_noise_bvp,
)


def test_encoder_unsupervised_output_shape():
    enc = AutoFiBVPEncoder(hidden_states=256)
    x = torch.randn(4, 22, 20, 20)
    out = enc(x)
    assert out.shape == (4, 256)


def test_encoder_supervised_returns_flat_features():
    enc = AutoFiBVPEncoder(hidden_states=256)
    x = torch.randn(4, 22, 20, 20)
    out = enc(x, flag="supervised")
    assert out.shape == (4, enc.feat_dim)


def test_parallel_unsupervised_returns_two_features():
    model = AutoFiParallel(num_classes=22, hidden_states=256)
    x1 = torch.randn(4, 22, 20, 20)
    x2 = torch.randn(4, 22, 20, 20)
    a, b = model(x1, x2)
    assert a.shape == (4, 256)
    assert b.shape == (4, 256)


def test_parallel_supervised_returns_logits():
    model = AutoFiParallel(num_classes=22, hidden_states=256)
    x = torch.randn(4, 22, 20, 20)
    y1, y2 = model(x, x, flag="supervised")
    assert y1.shape == (4, 22)
    assert y2.shape == (4, 22)


def test_gss_loss_components_finite():
    loss = AutoFiGSSLoss(tau=1.0, lam1=0.0, lam2=0.5)
    feat1 = torch.randn(8, 256)
    feat2 = torch.randn(8, 256)
    out = loss(feat1, feat2)
    for key in ("kl", "eh", "he", "kde", "final", "final-kde"):
        assert torch.isfinite(out[key]).all(), f"{key} is not finite"


def test_gss_loss_kl_is_zero_for_identical_inputs():
    loss = AutoFiGSSLoss(tau=1.0, eps=1e-12)
    feat = torch.randn(8, 16)
    out = loss(feat, feat)
    assert float(out["kl"].abs().item()) < 1e-6


def test_gaussian_noise_bvp_changes_input():
    x = torch.zeros(4, 22, 20, 20)
    y = gaussian_noise_bvp(x, epsilon=1.0)
    assert (y != x).any()
    assert y.shape == x.shape


def test_pretrain_autofi_runs_one_epoch():
    """Regression: pretrain_autofi previously crashed on torch.empty(generator=...).

    Exercises the actual RNG-sampling path inside the training loop.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from src.slices.josiah.autofi import AutoFiParallel, pretrain_autofi

    torch.manual_seed(0)
    x = torch.randn(8, 22, 20, 20)
    y = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = AutoFiParallel(num_classes=22, hidden_states=64)
    history = pretrain_autofi(model, loader, epochs=1, device="cpu", log_every=0)
    assert len(history) == 1
    assert all(map(lambda v: v == v, history))  # finite
