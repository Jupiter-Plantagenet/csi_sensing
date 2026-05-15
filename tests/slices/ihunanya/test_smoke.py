"""Smoke test for Slice 4's T4.1 scaffold.

Confirms the SimCLR pipeline runs end-to-end on stub data. T4.5 will add
the static-component perturbation augmentation; this test stays useful
as the no-augmentation reference path even after that.
"""

from __future__ import annotations

import torch

from src.slices.ihunanya.encoder import TinyCNN, count_parameters
from src.slices.ihunanya.run import main
from src.slices.ihunanya.ssl import SimCLR, make_views, nt_xent_loss


def test_pipeline_runs() -> None:
    acc = main(seed=42, epochs=2, batch_size=4)
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"


def test_encoder_param_count() -> None:
    encoder = TinyCNN(in_channels=30 * 3, feature_dim=128)
    n = count_parameters(encoder)
    # Per docs/07-experiment-scaffold.md, target is ~50K with the default
    # (30 subcarriers × 3 antennas) input channels. ~40K is fine.
    assert 20_000 < n < 80_000, f"unexpected parameter count: {n}"


def test_nt_xent_finite() -> None:
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    assert torch.isfinite(loss)


def test_make_views_returns_encoder_shape() -> None:
    x = torch.randn(2, 100, 30, 3)
    v1, v2 = make_views(x, augment_fn=None)
    assert v1.shape == (2, 90, 100)
    assert v2.shape == (2, 90, 100)


def test_simclr_one_step_runs() -> None:
    encoder = TinyCNN(in_channels=30 * 3, feature_dim=128)
    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 100, 30, 3)
    v1, v2 = make_views(x, augment_fn=None)
    z1 = model(v1)
    z2 = model(v2)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
