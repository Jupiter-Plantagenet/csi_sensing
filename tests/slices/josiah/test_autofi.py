"""Tests for the AutoFi exact reproduction (T5.4)."""

from __future__ import annotations

import torch

from src.slices.josiah.autofi import (
    AutoFiCNN,
    AutoFiGSS,
    autofi_augment,
    autofi_total_loss,
    geometric_loss,
    mutual_info_loss,
    pretrain_autofi,
    probability_consistency_loss,
)
from src.slices.josiah.data import CSI_A, CSI_S, CSI_T, NUM_CLASSES, StubCSI


def test_autofi_cnn_forward_shape() -> None:
    enc = AutoFiCNN(in_channels=CSI_A, s=CSI_S, t=CSI_T, feature_dim=128)
    x = torch.randn(2, CSI_A, CSI_S, CSI_T)
    out = enc(x)
    assert out.shape == (2, 128)


def test_autofi_gss_returns_two_softmax_distributions() -> None:
    gss = AutoFiGSS(in_channels=CSI_A, s=CSI_S, t=CSI_T, num_bins=NUM_CLASSES)
    x1 = torch.randn(4, CSI_A, CSI_S, CSI_T)
    x2 = torch.randn(4, CSI_A, CSI_S, CSI_T)
    p1, p2 = gss(x1, x2)
    assert p1.shape == p2.shape == (4, NUM_CLASSES)
    assert torch.allclose(p1.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(p2.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_probability_consistency_zero_for_identical_distributions() -> None:
    p = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    lp = probability_consistency_loss(p, p)
    assert lp.item() < 1e-6


def test_probability_consistency_positive_for_different_distributions() -> None:
    p1 = torch.tensor([[0.9, 0.05, 0.05]])
    p2 = torch.tensor([[0.05, 0.05, 0.9]])
    assert probability_consistency_loss(p1, p2).item() > 1.0


def test_mutual_info_loss_runs() -> None:
    p1 = torch.softmax(torch.randn(8, NUM_CLASSES), dim=-1)
    p2 = torch.softmax(torch.randn(8, NUM_CLASSES), dim=-1)
    lm = mutual_info_loss(p1, p2)
    assert lm.dim() == 0 and torch.isfinite(lm)


def test_geometric_loss_zero_for_identical_views() -> None:
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(6, NUM_CLASSES), dim=-1)
    lg = geometric_loss(p, p)
    assert lg.item() < 1e-5


def test_autofi_total_loss_returns_scalar_and_parts() -> None:
    p1 = torch.softmax(torch.randn(4, NUM_CLASSES), dim=-1)
    p2 = torch.softmax(torch.randn(4, NUM_CLASSES), dim=-1)
    total, parts = autofi_total_loss(p1, p2, lam=1.0, gamma=1000.0)
    assert total.dim() == 0
    assert set(parts) == {"L_p", "L_m", "L_g"}


def test_autofi_augment_adds_gaussian_noise() -> None:
    torch.manual_seed(0)
    x = torch.zeros(2, 10, CSI_S, CSI_A)
    out = autofi_augment(x, sigma=0.1)
    assert out.shape == x.shape
    assert out.abs().mean().item() > 0  # noise added


def test_pretrain_autofi_smoke() -> None:
    """Two epochs of GSS training on stub data — verifies the loop runs."""
    torch.manual_seed(0)
    ds = StubCSI(num_samples=8, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    gss = AutoFiGSS(in_channels=CSI_A, s=CSI_S, t=CSI_T, num_bins=NUM_CLASSES)
    losses = pretrain_autofi(gss, loader, epochs=2, lr=0.01, momentum=0.9)
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
