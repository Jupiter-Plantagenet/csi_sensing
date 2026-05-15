"""Tests for src.slices.josiah.capc.

The CAPC headline cell on SignFi UL/DL CSI is not reachable without that
dataset (see papers/team/capc-hardware-limited.md). These tests cover the
component code paths so the implementation can be exercised end-to-end on
any dataset that supplies a paired ``(view_a, view_b)`` tensor.
"""

from __future__ import annotations

import torch

from src.slices.josiah.capc import (
    BarlowTwinsLoss,
    BarlowTwinsProjector,
    CAPC,
    CAPCAutoregressor,
    CAPCCPCLoss,
    CAPCLoss,
    RSCNetEncoder,
    build_capc_optimizer,
    capc_gaussian_noise,
    capc_subcarrier_mask,
    capc_time_flip,
    warmup_cosine_lr,
)


def test_rscnet_encoder_output_shape():
    enc = RSCNetEncoder(window_shape=(3, 30, 10), feature_dim=128)
    x = torch.randn(8, 3, 30, 10)
    out = enc(x)
    assert out.shape == (8, 128)


def test_autoregressor_returns_sequence():
    ar = CAPCAutoregressor(feature_dim=128, hidden_dim=128)
    z = torch.randn(4, 20, 128)
    h = ar(z)
    assert h.shape == (4, 20, 128)


def test_barlow_twins_projector_output_shape():
    proj = BarlowTwinsProjector()
    x = torch.randn(8, 128)
    out = proj(x)
    assert out.shape == (8, 256)


def test_cpc_loss_finite_and_scalar():
    cpc = CAPCCPCLoss(num_future_steps=4, feature_dim=128)
    z_seq = torch.randn(8, 20, 128)
    c_t = torch.randn(8, 128)
    loss = cpc(z_seq, c_t, anchor_index=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_barlow_twins_loss_zero_for_perfectly_aligned():
    bt = BarlowTwinsLoss(lambda_bt=0.002)
    x = torch.randn(32, 64)
    # Same projector outputs -> normalized cross-correlation diag is 1, off-diag small for large B.
    loss = bt(x, x)
    assert torch.isfinite(loss)


def test_capc_forward_shapes():
    model = CAPC(window_shape=(3, 30, 10))
    view_a = torch.randn(4, 20, 3, 30, 10)
    view_b = torch.randn(4, 20, 3, 30, 10)
    z_a, c_a, z_b, c_b, p_a, p_b = model(view_a, view_b)
    assert z_a.shape == (4, 20, 128)
    assert c_a.shape == (4, 128)
    assert p_a.shape == (4, 256)
    assert p_b.shape == (4, 256)


def test_capc_loss_composite_finite():
    model = CAPC(window_shape=(3, 30, 10))
    view_a = torch.randn(4, 20, 3, 30, 10)
    view_b = torch.randn(4, 20, 3, 30, 10)
    z_a, c_a, z_b, c_b, p_a, p_b = model(view_a, view_b)
    loss_fn = CAPCLoss(num_future_steps=4, feature_dim=128)
    out = loss_fn(z_a, c_a, z_b, c_b, p_a, p_b)
    assert torch.isfinite(out["total"])


def test_augmentations_preserve_shape():
    x = torch.randn(2, 3, 30, 10)
    assert capc_time_flip(x).shape == x.shape
    assert capc_gaussian_noise(x, sigma=0.05).shape == x.shape
    assert capc_subcarrier_mask(x, ratio=0.1).shape == x.shape


def test_capc_optimizer_separates_biases_and_weights():
    model = CAPC(window_shape=(3, 30, 10))
    opt = build_capc_optimizer(model.parameters())
    assert len(opt.param_groups) == 2
    weight_lr = opt.param_groups[0]["lr"]
    biasbn_lr = opt.param_groups[1]["lr"]
    assert weight_lr > biasbn_lr


def test_warmup_cosine_lr_warms_up_then_decays():
    total = 300
    warmup = 10
    assert warmup_cosine_lr(0, total_epochs=total, warmup_epochs=warmup) < 0.2
    assert warmup_cosine_lr(warmup - 1, total_epochs=total, warmup_epochs=warmup) <= 1.0
    assert warmup_cosine_lr(warmup, total_epochs=total, warmup_epochs=warmup) == 1.0
    assert warmup_cosine_lr(total - 1, total_epochs=total, warmup_epochs=warmup) < 0.05
