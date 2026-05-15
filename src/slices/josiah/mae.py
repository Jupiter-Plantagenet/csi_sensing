"""Masked Autoencoder (MAE) for Widar3.0 BVP — published-baseline reproduction
target for Era 5 of the team's literature survey.

Reference papers:

* He, Chen, Xie, Li, Doll a r, Girshick. *Masked Autoencoders Are Scalable
  Vision Learners.* CVPR 2022. arXiv:2111.06377.
* Xu et al. *Evaluating SSL for WiFi CSI-Based HAR: A Systematic Study.*
  ACM TOSN 2025. (SSLCSI). MAE-on-Widar cell.

The team paper differentiates against "augmentation abandoned via masking"
(CIG-MAE / SSLCSI-MAE) — see ``papers/team/baselines-figure.md`` and the
literature survey slide 17. This module provides the comparable row.

Input representation: Widar3.0 BVP ``(T=22, vx=20, vy=20)`` from
``src/slices/josiah/widar_bvp.py``. Each time step is a single token (a
``vx*vy = 400``-dim spatial slice) so the masking unit is "one full
velocity-grid snapshot at one time index", which is the natural temporal
analog of MAE's ``16x16`` image patches.

Architecture defaults are kept modest to fit the 24k-sample dataset:

* ``emb_dim = 128``  (matches our project SimCLR/AutoFi feature dim)
* ``encoder_depth = 4``, ``decoder_depth = 2``, ``num_heads = 4``
* ``mask_ratio = 0.75`` (paper default).

Training:

* SSL pre-train: AdamW, base LR ``1.5e-4`` with 40-epoch warmup, cosine
  decay, weight decay ``0.05``, 200 epochs total. Loss: MSE on masked-token
  positions only (paper Eq. 1).
* Linear probe: Adam ``1e-3`` on frozen encoder features (mean-pool of
  encoder outputs), 100 epochs. Matches our project-baseline probe.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .widar_bvp import BVP_T, BVP_VX, BVP_VY, NUM_BVP_CLASSES

NUM_PROJECT_CLASSES = 6


# -----------------------------------------------------------------------------
# Patch / token embedding.


class BVPTokenEmbed(nn.Module):
    """Each time-step's ``(vx, vy)`` spatial map becomes one token.

    Input: ``(B, T, vx, vy)``. Output: ``(B, T, emb_dim)``.
    """

    def __init__(self, *, vx: int = BVP_VX, vy: int = BVP_VY, emb_dim: int = 128) -> None:
        super().__init__()
        self.vx = vx
        self.vy = vy
        self.emb_dim = emb_dim
        self.proj = nn.Linear(vx * vy, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, vx, vy = x.shape
        return self.proj(x.reshape(b, t, vx * vy))


# -----------------------------------------------------------------------------
# Transformer block (paper-faithful pre-norm transformer).


class _TransformerBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def _sinusoidal_positional_embedding(num_positions: int, dim: int) -> torch.Tensor:
    """Standard sin/cos positional embedding from Vaswani et al. 2017."""
    pe = torch.zeros(num_positions, dim)
    position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


# -----------------------------------------------------------------------------
# MAE encoder.


class MAEEncoder(nn.Module):
    """Encode visible tokens with positional embeddings (paper §3.2)."""

    def __init__(
        self,
        *,
        num_tokens: int = BVP_T,
        token_dim: int = BVP_VX * BVP_VY,
        emb_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.embed = BVPTokenEmbed(vx=BVP_VX, vy=BVP_VY, emb_dim=emb_dim)
        self.register_buffer(
            "pos_embed",
            _sinusoidal_positional_embedding(num_tokens, emb_dim).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim=emb_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward used by the linear probe and finetuning.

        ``x``: ``(B, T, vx, vy)``. Returns ``(B, T, emb_dim)``.
        """
        z = self.embed(x) + self.pos_embed
        for block in self.blocks:
            z = block(z)
        return self.norm(z)

    def forward_masked(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random per-sample masking, encode visible tokens only.

        Returns ``(z_visible, mask, ids_restore)`` where:
        * ``z_visible``: ``(B, T_visible, emb_dim)`` -- encoded visible tokens.
        * ``mask``: ``(B, T)`` binary, 1 = masked, 0 = kept.
        * ``ids_restore``: ``(B, T)`` permutation that restores the original
          order when applied to the concatenation of ``z_visible`` and mask
          tokens.
        """
        b, t, _, _ = x.shape
        emb = self.embed(x) + self.pos_embed  # (B, T, D)

        n_keep = max(1, int(t * (1.0 - mask_ratio)))
        noise = torch.rand(b, t, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        ids_restore = ids_shuffle.argsort(dim=1)

        z_visible = torch.gather(
            emb, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, emb.shape[-1])
        )
        for block in self.blocks:
            z_visible = block(z_visible)
        z_visible = self.norm(z_visible)

        mask = torch.ones(b, t, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)
        return z_visible, mask, ids_restore


# -----------------------------------------------------------------------------
# MAE decoder.


class MAEDecoder(nn.Module):
    """Lightweight decoder that reconstructs the original token at each position."""

    def __init__(
        self,
        *,
        num_tokens: int = BVP_T,
        token_dim: int = BVP_VX * BVP_VY,
        emb_dim: int = 128,
        decoder_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj_in = nn.Linear(emb_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed",
            _sinusoidal_positional_embedding(num_tokens, decoder_dim).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim=decoder_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.proj_out = nn.Linear(decoder_dim, token_dim)

    def forward(
        self, z_visible: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct all tokens.

        Returns ``(B, T, token_dim)``.
        """
        b, n_visible, _ = z_visible.shape
        x = self.proj_in(z_visible)
        n_total = ids_restore.shape[1]
        n_masked = n_total - n_visible
        mask_tokens = self.mask_token.expand(b, n_masked, -1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # (B, T, D)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1])
        )
        x_ = x_ + self.pos_embed
        for block in self.blocks:
            x_ = block(x_)
        x_ = self.norm(x_)
        return self.proj_out(x_)


# -----------------------------------------------------------------------------
# Full MAE model.


class BVPMAE(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int = 128,
        encoder_depth: int = 4,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        num_heads: int = 4,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = MAEEncoder(
            emb_dim=emb_dim, depth=encoder_depth, num_heads=num_heads
        )
        self.decoder = MAEDecoder(
            emb_dim=emb_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """``x``: ``(B, T, vx, vy)``."""
        b, t, vx, vy = x.shape
        target = x.reshape(b, t, vx * vy)
        z_visible, mask, ids_restore = self.encoder.forward_masked(x, self.mask_ratio)
        pred = self.decoder(z_visible, ids_restore)
        # MSE on masked positions only.
        loss_full = (pred - target).pow(2).mean(dim=-1)
        loss = (loss_full * mask).sum() / mask.sum().clamp(min=1.0)
        return {"loss": loss, "pred": pred, "mask": mask}


# -----------------------------------------------------------------------------
# Training loops.


def _warmup_cosine_lr(epoch: int, *, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    p = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def pretrain_mae(
    model: BVPMAE,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 40,
    device: str = "cpu",
    log_every: int = 10,
) -> list[float]:
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    for epoch in range(epochs):
        # Warmup + cosine schedule, multiplier of base LR.
        mult = _warmup_cosine_lr(epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)
        for g in optim.param_groups:
            g["lr"] = lr * mult
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            out = model(x)
            loss = out["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
            n += 1
        history.append(total / max(1, n))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[mae-ssl] epoch {epoch+1}/{epochs} mse={history[-1]:.4f} lr={lr*mult:.2e}")
    return history


@torch.no_grad()
def _extract_mae_features(
    encoder: MAEEncoder, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval().to(device)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device).float()
        z = encoder.forward_features(x)  # (B, T, D)
        pooled = z.mean(dim=1)
        feats.append(pooled.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def linear_probe_mae(
    encoder: MAEEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str = "cpu",
    max_iter: int = 1000,
    seed: int = 42,
) -> float:
    tr_x, tr_y = _extract_mae_features(encoder, train_loader, device)
    te_x, te_y = _extract_mae_features(encoder, test_loader, device)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed).fit(tr_x, tr_y)
    return float(np.mean(clf.predict(te_x) == te_y))


# -----------------------------------------------------------------------------
# Runner entry point.


def run_mae(
    *,
    seed: int,
    epochs: int = 200,
    batch_size: int = 64,
    bvp_root: str = "data/widar3/Widardata",
    cache_dir: str = "data/widar3/cache",
    gestures: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    num_classes: int = NUM_PROJECT_CLASSES,
    mask_ratio: float = 0.75,
    emb_dim: int = 128,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 4,
) -> float:
    """Single-seed MAE run on Widar BVP cross-subject. Returns linear-probe acc."""
    import random
    from .widar_bvp import WidarBVP

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gestures_list = list(gestures)
    cache_tag = f"josiah-bvp-cs-g{'-'.join(map(str, gestures_list))}"
    train_ds = WidarBVP(
        root=bvp_root,
        split="cross-subject",
        train=True,
        gesture_filter=gestures_list,
        cache_path=f"{cache_dir}/{cache_tag}-train.pt",
    )
    test_ds = WidarBVP(
        root=bvp_root,
        split="cross-subject",
        train=False,
        gesture_filter=gestures_list,
        cache_path=f"{cache_dir}/{cache_tag}-test.pt",
    )
    print(f"[mae] train={len(train_ds)} test={len(test_ds)} mask_ratio={mask_ratio}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BVPMAE(
        emb_dim=emb_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mask_ratio=mask_ratio,
    )
    pretrain_mae(model, train_loader, epochs=epochs, device=device)
    acc = linear_probe_mae(
        model.encoder, probe_train_loader, test_loader, device=device, seed=seed
    )
    print(f"[mae] linear-probe acc={acc:.4f}")
    return acc
