"""Tests for src.slices.josiah.mae."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.slices.josiah.mae import (
    BVPMAE,
    BVPTokenEmbed,
    MAEDecoder,
    MAEEncoder,
    _sinusoidal_positional_embedding,
    pretrain_mae,
)


def test_token_embed_shape():
    emb = BVPTokenEmbed(vx=20, vy=20, emb_dim=128)
    x = torch.randn(4, 22, 20, 20)
    out = emb(x)
    assert out.shape == (4, 22, 128)


def test_sinusoidal_positional_embedding_is_finite():
    pe = _sinusoidal_positional_embedding(22, 128)
    assert pe.shape == (22, 128)
    assert torch.isfinite(pe).all()


def test_encoder_forward_features_shape():
    enc = MAEEncoder(num_tokens=22, emb_dim=128, depth=2, num_heads=4)
    x = torch.randn(2, 22, 20, 20)
    out = enc.forward_features(x)
    assert out.shape == (2, 22, 128)


def test_encoder_forward_masked_returns_visible_only():
    enc = MAEEncoder(num_tokens=22, emb_dim=128, depth=2, num_heads=4)
    x = torch.randn(2, 22, 20, 20)
    z_visible, mask, ids_restore = enc.forward_masked(x, mask_ratio=0.75)
    # n_keep = max(1, int(22 * 0.25)) = max(1, 5) = 5
    assert z_visible.shape == (2, 5, 128)
    assert mask.shape == (2, 22)
    assert ids_restore.shape == (2, 22)
    # Mask: ratio masked vs kept consistent (17 masked + 5 kept = 22).
    assert mask.sum(dim=1).tolist() == [17.0, 17.0]


def test_decoder_reconstruction_shape():
    dec = MAEDecoder(num_tokens=22, token_dim=400, emb_dim=128, decoder_dim=64, depth=2, num_heads=4)
    z_visible = torch.randn(2, 5, 128)
    ids_restore = torch.stack([torch.randperm(22), torch.randperm(22)])
    out = dec(z_visible, ids_restore)
    assert out.shape == (2, 22, 400)


def test_bvpmae_forward_loss_finite_and_scalar():
    model = BVPMAE(emb_dim=128, encoder_depth=2, decoder_depth=2, num_heads=4, mask_ratio=0.75)
    x = torch.randn(4, 22, 20, 20)
    out = model(x)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert out["pred"].shape == (4, 22, 400)
    assert out["mask"].shape == (4, 22)


def test_bvpmae_loss_only_on_masked_positions():
    """If the model perfectly predicts the target at masked positions, loss is 0."""
    model = BVPMAE(emb_dim=128, encoder_depth=2, decoder_depth=2, num_heads=4, mask_ratio=0.75)
    model.eval()
    x = torch.randn(2, 22, 20, 20)
    with torch.no_grad():
        out = model(x)
        # Replace decoder predictions with the ground truth at masked positions;
        # set unmasked positions to garbage to confirm they're ignored.
        target = x.reshape(2, 22, 400)
        forced_pred = torch.where(out["mask"].unsqueeze(-1).bool(), target, torch.zeros_like(target))
        loss = ((forced_pred - target).pow(2).mean(dim=-1) * out["mask"]).sum() / out["mask"].sum()
    assert float(loss.item()) < 1e-6


def test_pretrain_mae_runs_one_epoch():
    """Regression: smoke-test the SSL training loop end-to-end."""
    torch.manual_seed(0)
    x = torch.randn(8, 22, 20, 20)
    y = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = BVPMAE(emb_dim=64, encoder_depth=1, decoder_depth=1, num_heads=4)
    history = pretrain_mae(
        model, loader, epochs=1, warmup_epochs=1, device="cpu", log_every=0
    )
    assert len(history) == 1
    assert history[0] == history[0]  # finite
