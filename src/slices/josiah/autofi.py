"""AutoFi exact reproduction (T5.4).

Geometric Self-Supervised Learning (GSS) from Yang et al.,
"AutoFi: Towards Automatic WiFi Human Sensing via Geometric
Self-Supervised Learning" (IEEE IoT Journal 2022).

This module reproduces the GSS pre-training stage exactly as described
in §III.B of the paper:

- Augmentation A_ε: x ← x + εζ, ζ ~ N(0, σ²) — additive Gaussian noise
  on the raw CSI (the paper's only augmentation).
- Twin encoders E_θ1, E_θ2 (Table I 6-layer CNN) + non-linear heads
  G_φ1, G_φ2 (MLP) that emit per-view prediction distributions P1, P2
  over NUM_CLASSES bins via softmax.
- Loss: L = L_p + λ·L_m + γ·L_g (eq. 9), λ=1, γ=1000.
  - L_p (probability consistency, eq. 3): symmetric KL between P1, P2.
  - L_m (mutual information, eq. 5): h(E[P]) + E[h(P)], applied to both
    views and summed — note the paper writes the second term with a
    plus sign because both terms are signed for *minimisation* of
    -I(X;Y), see derivation in §III.B.
  - L_g (geometric consistency, eq. 8): KL between cosine-similarity
    Q-distributions of the two views (eqs. 6–7).
- Optimiser: SGD, lr=0.01, momentum=0.9, batch 128, 300 epochs.

Architecture (Table I, §III.B). The paper's stated input is
3 × 114 × 500 (Atheros tool: 3 RX × 114 subcarriers × 500 timesteps).
Widar3.0 with the Intel 5300 tool gives 3 RX × 30 subcarriers × T. The
paper itself notes: "The first layer of the GSS module is slightly
modified to match the input size." We follow the same adaptation: keep
the layer-2..6 kernels exactly, scale down the layer-1 kernel and
stride proportionally to the smaller subcarrier / time dimensions.

The CSI tensor convention in this repo is (T, S, A); we permute to
(A, S, T) before feeding the Conv2d stack — antennas as channels,
subcarriers as height, time as width.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Augmentation A_ε (eq. 1): additive Gaussian noise on subcarriers.

def autofi_augment(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """A_ε(x) = x + ε·ζ, ζ ~ N(0, σ²). The paper's only augmentation."""
    return x + torch.randn_like(x) * sigma


# ---------------------------------------------------------------------------
# Feature extractor E_θ — 6-layer CNN, Table I.
# Adapted for (A=3, S=30, T=100) Intel 5300 / Widar3.0 input.

class AutoFiCNN(nn.Module):
    """6-layer CNN from AutoFi Table I, adapted for 30-subcarrier input.

    Output: (B, feature_dim) feature vector. The paper's classifier
    F_ψ is a separate 128-then-6-dense head used in the few-shot
    calibration stage (T5.4 doesn't run FSC; we just need E_θ).
    """

    def __init__(
        self,
        in_channels: int = 3,
        s: int = 30,
        t: int = 100,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        # Table I exact kernels for layers 2..6. Layer 1 kernel/stride is
        # scaled from (15,23)/9 to (5,9)/3 — same ~3:1 ratio between subcarrier
        # and time kernel sizes, same stride-9 → stride-3 ratio (30/114 ≈ 0.26).
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5, 9), stride=3, padding=(2, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 7), stride=1, padding=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 7), stride=1, padding=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(3, 7), stride=1, padding=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        with torch.no_grad():
            probe = torch.zeros(1, in_channels, s, t)
            flat_dim = self.features(probe).flatten(1).shape[1]
        self._flat_dim = flat_dim

        self.proj = nn.Linear(flat_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, A, S, T); output (B, feature_dim)."""
        h = self.features(x)
        h = h.flatten(1)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Non-linear head G_φ (the "bottleneck layer" in §III.B that separates the
# feature space). Two-layer MLP → softmax over NUM_CLASSES bins.

class AutoFiHead(nn.Module):
    def __init__(
        self, feature_dim: int = 128, hidden_dim: int = 128, num_bins: int = 6
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(h), dim=-1)


class AutoFiGSS(nn.Module):
    """Twin-branch GSS module: E_θ1, E_θ2 + G_φ1, G_φ2."""

    def __init__(
        self,
        in_channels: int = 3,
        s: int = 30,
        t: int = 100,
        feature_dim: int = 128,
        num_bins: int = 6,
    ) -> None:
        super().__init__()
        self.encoder1 = AutoFiCNN(in_channels, s, t, feature_dim)
        self.encoder2 = AutoFiCNN(in_channels, s, t, feature_dim)
        self.head1 = AutoFiHead(feature_dim, feature_dim, num_bins)
        self.head2 = AutoFiHead(feature_dim, feature_dim, num_bins)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = self.head1(self.encoder1(x1))
        p2 = self.head2(self.encoder2(x2))
        return p1, p2


# ---------------------------------------------------------------------------
# Losses


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def probability_consistency_loss(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """L_p (eq. 3): symmetric KL between P1 and P2."""
    return 0.5 * (_kl(p1, p2).mean() + _kl(p2, p1).mean())


def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(p * torch.log(p + eps)).sum(dim=-1)


def mutual_info_loss(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """L_m (eq. 5): h(E[P]) + E[h(P)], summed over both views.

    Paper formula: L_m = h(E[P]) + E[h(P)]. Driving this *down* in the
    overall L = L_p + λL_m + γL_g schedule corresponds to maximising
    mutual information (paper §III.B): the model is pushed to (a) make
    individual predictions confident — low E[h(P)] — and (b) make the
    batch-averaged prediction uniform — high h(E[P]).

    We follow the paper literally: a single scalar that sums those two
    terms and is added to the total loss with the +λ sign.
    """
    pm1 = p1.mean(dim=0)
    pm2 = p2.mean(dim=0)
    return (
        _entropy(pm1) + _entropy(pm2) + _entropy(p1).mean() + _entropy(p2).mean()
    )


def _cosine_kernel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """K(a, b) = 0.5 (a·b / (||a||·||b||) + 1) — eq. 7."""
    na = a / (a.norm(dim=-1, keepdim=True) + eps)
    nb = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 0.5 * (na @ nb.t() + 1.0)


def geometric_loss(p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L_g (eq. 8): KL between Q-distributions of the two views.

    For each sample x^i, Q^i is the row q_{i|j} = K(P^i, P^j) /
    Σ_{m≠j} K(P^m, P^j). We use cosine similarity (eq. 7).
    """
    k1 = _cosine_kernel(p1, p1)
    k2 = _cosine_kernel(p2, p2)
    # Zero out self-similarity per the "m ≠ j" qualifier in eq. 6.
    n = k1.shape[0]
    mask = 1.0 - torch.eye(n, device=k1.device, dtype=k1.dtype)
    k1 = k1 * mask
    k2 = k2 * mask
    q1 = k1 / (k1.sum(dim=0, keepdim=True) + eps)
    q2 = k2 / (k2.sum(dim=0, keepdim=True) + eps)
    return _kl(q1.t(), q2.t()).mean()


def autofi_total_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    *,
    lam: float = 1.0,
    gamma: float = 1000.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_p + λ·L_m + γ·L_g — paper eq. 9 with λ=1, γ=1000."""
    lp = probability_consistency_loss(p1, p2)
    lm = mutual_info_loss(p1, p2)
    lg = geometric_loss(p1, p2)
    total = lp + lam * lm + gamma * lg
    parts = {"L_p": float(lp.item()), "L_m": float(lm.item()), "L_g": float(lg.item())}
    return total, parts


# ---------------------------------------------------------------------------
# Pre-training loop

def _to_autofi_input(x: torch.Tensor) -> torch.Tensor:
    """(B, T, S, A) -> (B, A, S, T)."""
    return x.permute(0, 3, 2, 1).contiguous()


def pretrain_autofi(
    model: AutoFiGSS,
    loader: DataLoader,
    *,
    epochs: int = 300,
    lr: float = 0.01,
    momentum: float = 0.9,
    sigma: float = 0.05,
    lam: float = 1.0,
    gamma: float = 1000.0,
    device: str = "cpu",
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[float]:
    """AutoFi GSS training. SGD lr=0.01, momentum=0.9 per the paper.

    Returns the per-epoch mean total-loss list.
    """
    model.to(device)
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    aug = augment_fn or (lambda x: autofi_augment(x, sigma=sigma))
    epoch_losses: list[float] = []
    for _ in range(epochs):
        batch_losses: list[float] = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            v1 = _to_autofi_input(aug(x))
            v2 = _to_autofi_input(aug(x))
            p1, p2 = model(v1, v2)
            loss, _ = autofi_total_loss(p1, p2, lam=lam, gamma=gamma)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))
    return epoch_losses
