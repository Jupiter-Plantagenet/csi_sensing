"""Unit tests for Slice 5's SSL machinery (T5.3).

Covers NT-Xent shape/finiteness, `make_views` randomness, `random_crop`
preserves shape, and a one-batch SimCLR step doesn't blow up.
"""

from __future__ import annotations

import torch

from src.slices.josiah.augmentations import random_crop
from src.slices.josiah.encoder import TinyCNN
from src.slices.josiah.ssl import SimCLR, make_views, nt_xent_loss


def test_nt_xent_finite() -> None:
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_random_crop_preserves_shape() -> None:
    x = torch.randn(2, 100, 30, 3)
    out = random_crop(x, crop_ratio=0.7)
    assert out.shape == x.shape
    # The last ~30% of the time axis should be zeros (padding region).
    # Exact zero region depends on the random start; just check sparsity:
    zero_ratio = (out == 0).float().mean().item()
    assert zero_ratio > 0.0


def test_random_crop_two_views_differ() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 100, 30, 3)
    v1 = random_crop(x, crop_ratio=0.7)
    v2 = random_crop(x, crop_ratio=0.7)
    # Two independent calls should produce two different starts -> different outputs.
    # With 30 possible start positions, the collision probability is ~1/30,
    # so this assertion is robust under fixed seed.
    assert not torch.equal(v1, v2)


def test_make_views_returns_encoder_shape() -> None:
    x = torch.randn(2, 100, 30, 3)
    v1, v2 = make_views(x, augment_fn=random_crop)
    # Encoder expects (B, S*A, T) = (2, 90, 100).
    assert v1.shape == (2, 90, 100)
    assert v2.shape == (2, 90, 100)


def test_simclr_one_step_runs() -> None:
    encoder = TinyCNN(in_channels=30 * 3, feature_dim=128)
    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 100, 30, 3)
    v1, v2 = make_views(x, augment_fn=random_crop)
    z1 = model(v1)
    z2 = model(v2)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
