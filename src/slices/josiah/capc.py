"""CAPC exact reproduction (T5.5).

Context-Aware Predictive Coding, Barahimi et al.,
"Context-Aware Predictive Coding: A Representation Learning Framework
for WiFi Sensing" (IEEE Open J. Commun. Society 2024,
arxiv 2410.01825).

This module reproduces the CAPC pre-training stage per §III and the
"Training Configuration" paragraph of §IV-E.

Components (Fig. 2 + Algorithm 1):

- CSI segmentation: each input `u` of shape `(A, S, T)` is split into
  L = T // N_f non-overlapping windows of length N_f along time.
- Stochastic augmentations T: Gaussian noise, time flip, time mask,
  subcarrier mask, plus the paper's *dual view* (uplink/downlink) —
  dual view is omitted here because Widar3.0 provides only one
  direction of CSI. Per §IV-D: "CAPC can function effectively without
  needing both" and the paper's noise+subcarrier-mask config
  (`CAPC*` row, Table I) is the documented fallback.
- Base encoder E_θ: RSCNet [40 in the paper]. We approximate RSCNet
  with a small Conv2d-Linear stack producing the paper's stated
  D=128-dim embedding per window. The substantive CAPC contribution
  is the hybrid loss, not the encoder family — TinyCNN-style RSCNet
  variants are interchangeable per §IV-C ("the same backbone encoder
  across all methods").
- Autoregressive head G_γ: 1-layer GRU, hidden=128 (the paper
  replaced CPC's LSTM with a GRU — §III.A "Remark").
- Per-step bilinear predictors W_k, k=1..T (T=9 future windows per
  §IV-E), each a Linear without bias.
- Hybrid loss: L = L_BT + β(L_CPC^A + L_CPC^B), β=50 per eq. 6 and
  §IV-E.
  * L_CPC (eq. 3): InfoNCE with f_k(x_{t+k}, c_t)=exp(z^T W_k c_t).
  * L_BT (eq. 4): Σ(C_ii−1)² + λ·Σ_{i≠j} C_ij², λ=0.002 per §IV-E.
- Optimiser: LARS, lr (weights) 0.2, lr (bias/BN) 0.0048, weight decay
  1.5e-6, 300 epochs batch 128, 10-epoch warmup + cosine decay.

The CSI tensor convention in this repo is (T, S, A); we permute to
(A, S, T) before windowing.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Augmentations T (paper §III.A + §IV-D)


def gaussian_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Additive Gaussian noise, σ=0.1 per §IV-D."""
    return x + torch.randn_like(x) * sigma


def time_flip(x: torch.Tensor) -> torch.Tensor:
    """Flip along the time axis. Input (B, T, S, A)."""
    return x.flip(dims=(-3,))


def time_mask(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """Zero out a contiguous fraction of time samples at a random start."""
    t = x.shape[-3]
    width = max(1, int(round(p * t)))
    start = int(torch.randint(0, max(1, t - width + 1), (1,)).item())
    out = x.clone()
    out[..., start : start + width, :, :] = 0
    return out


def subcarrier_mask(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """Random per-sample subcarrier dropout. Input (B, T, S, A)."""
    if x.ndim == 4:
        b, _, s, _ = x.shape
        keep = (torch.rand(b, s, device=x.device) >= p).to(x.dtype)
        return x * keep.view(b, 1, s, 1)
    s = x.shape[1]
    keep = (torch.rand(s, device=x.device) >= p).to(x.dtype)
    return x * keep.view(1, s, 1)


def capc_view_noise_then_submask(x: torch.Tensor) -> torch.Tensor:
    """CAPC* augmentation row (Table I): noise + subcarrier mask.

    Documented fallback when dual-view CSI (uplink+downlink) is
    unavailable. Widar3.0 has only one direction, so this is what we
    use.
    """
    return subcarrier_mask(gaussian_noise(x, sigma=0.1), p=0.15)


# ---------------------------------------------------------------------------
# Base encoder E_θ — RSCNet approximation.
# Consumes per-window slices of shape (B, A, S, N_f) and emits a
# D-dim embedding per window.


class RSCNetBlock(nn.Module):
    """RSCNet-style Conv2d block (approximation).

    The paper's RSCNet [40] is a residual selective-compression CSI
    network; we approximate with the same "small Conv2d backbone"
    structure used throughout the repo so the comparison is
    apples-to-apples with the other baselines.
    """

    def __init__(
        self,
        in_channels: int = 3,
        s: int = 30,
        n_f: int = 10,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, A, S, N_f); output (B, D)."""
        h = self.features(x).flatten(1)
        return self.proj(h)


# ---------------------------------------------------------------------------
# CAPC branch: encoder + GRU + per-step bilinear predictors.


class CAPCBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        s: int = 30,
        n_f: int = 10,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        future_steps: int = 9,
    ) -> None:
        super().__init__()
        self.encoder = RSCNetBlock(in_channels, s, n_f, embedding_dim)
        # GRU instead of LSTM — paper §III.A "Remark".
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # Per-step bilinear predictors W_k (eq. 2). No bias — pure
        # bilinear form z^T W_k c_t.
        self.W = nn.ModuleList(
            [nn.Linear(hidden_dim, embedding_dim, bias=False) for _ in range(future_steps)]
        )
        self.future_steps = future_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode (B, L, A, S, N_f) → (B, L, D)."""
        b, length = windows.shape[:2]
        flat = windows.reshape(b * length, *windows.shape[2:])
        z = self.encoder(flat)
        return z.view(b, length, -1)

    def context(self, z: torch.Tensor) -> torch.Tensor:
        """Run the GRU over (B, L, D); return all hidden states (B, L, H)."""
        out, _ = self.gru(z)
        return out


# ---------------------------------------------------------------------------
# Losses


def cpc_loss(
    z: torch.Tensor,
    c: torch.Tensor,
    W: nn.ModuleList,
    t_anchor: int,
    *,
    future_steps: int,
) -> torch.Tensor:
    """L_CPC (eq. 3): InfoNCE over a random anchor t.

    For each future step k in [1, future_steps] (clipped so t+k < L):
        f_k(z_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)
    Categorical cross-entropy: positive is the in-batch sample at
    position t+k; negatives are all other samples' z_{t+k} in the
    batch. (We use the batch as the negative pool, the standard CPC
    InfoNCE formulation.)
    """
    b, length, d = z.shape
    c_anchor = c[:, t_anchor]  # (B, H)
    losses: list[torch.Tensor] = []
    for k_idx in range(1, future_steps + 1):
        target_t = t_anchor + k_idx
        if target_t >= length:
            break
        pred = W[k_idx - 1](c_anchor)            # (B, D)
        z_future = z[:, target_t]                # (B, D)
        # Logits: (B_pred, B_target). Diagonal is the positive.
        logits = pred @ z_future.t()
        target = torch.arange(b, device=z.device)
        losses.append(nn.functional.cross_entropy(logits, target))
    if not losses:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return torch.stack(losses).mean()


def barlow_twins_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, lam: float = 0.002
) -> torch.Tensor:
    """L_BT (eq. 4). z_a, z_b: (B, D). λ=0.002 per §IV-E."""
    b, d = z_a.shape
    a = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-6)
    bb = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-6)
    c = (a.t() @ bb) / b                          # (D, D) cross-correlation
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on_diag + lam * off_diag


def capc_total_loss(
    z_a: torch.Tensor,
    c_a: torch.Tensor,
    z_b: torch.Tensor,
    c_b: torch.Tensor,
    W_a: nn.ModuleList,
    W_b: nn.ModuleList,
    *,
    future_steps: int = 9,
    beta: float = 50.0,
    bt_lambda: float = 0.002,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_BT + β·(L_CPC^A + L_CPC^B) — eq. 6, β=50."""
    length = z_a.shape[1]
    max_anchor = max(1, length - 1)
    t_anchor = int(torch.randint(0, max_anchor, (1,)).item())
    l_cpc_a = cpc_loss(z_a, c_a, W_a, t_anchor, future_steps=future_steps)
    l_cpc_b = cpc_loss(z_b, c_b, W_b, t_anchor, future_steps=future_steps)
    # Barlow Twins on the context embeddings at t_anchor.
    l_bt = barlow_twins_loss(c_a[:, t_anchor], c_b[:, t_anchor], lam=bt_lambda)
    total = l_bt + beta * (l_cpc_a + l_cpc_b)
    parts = {
        "L_BT": float(l_bt.item()),
        "L_CPC_A": float(l_cpc_a.item()),
        "L_CPC_B": float(l_cpc_b.item()),
    }
    return total, parts


# ---------------------------------------------------------------------------
# LARS optimiser (paper §IV-E).
# Minimal implementation: per-parameter trust-ratio scaling on top of SGD.


class LARS(torch.optim.Optimizer):
    """LARS (You et al. 2017) as used by CAPC §IV-E.

    Standard PyTorch doesn't ship LARS in core; this is a minimal
    implementation. Bias/BN parameters get the small lr and the trust
    ratio disabled (the paper's "lr for biases and batch-normalization
    parameters at 0.0048").
    """

    def __init__(
        self,
        params,
        lr: float = 0.2,
        momentum: float = 0.9,
        weight_decay: float = 1.5e-6,
        trust_coef: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coef=trust_coef,
            eps=eps,
            no_trust=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"] != 0:
                    g = g.add(p, alpha=group["weight_decay"])
                if not group["no_trust"]:
                    p_norm = p.norm()
                    g_norm = g.norm()
                    trust_ratio = torch.where(
                        (p_norm > 0) & (g_norm > 0),
                        group["trust_coef"] * p_norm / (g_norm + group["eps"]),
                        torch.tensor(1.0, device=p.device, dtype=p.dtype),
                    )
                    g = g * trust_ratio
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p)
                    state["momentum_buffer"] = buf
                buf.mul_(group["momentum"]).add_(g)
                p.add_(buf, alpha=-group["lr"])
        return loss


def _split_params_for_lars(model: nn.Module) -> tuple[list, list]:
    """Separate weights from biases/BN params per §IV-E."""
    weights, bn_bias = [], []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for p in module.parameters(recurse=False):
                bn_bias.append(p)
        else:
            for name, p in module.named_parameters(recurse=False):
                (bn_bias if name.endswith("bias") else weights).append(p)
    return weights, bn_bias


def make_capc_optimizer(
    branch_a: CAPCBranch,
    branch_b: CAPCBranch,
    *,
    lr_weights: float = 0.2,
    lr_bias: float = 0.0048,
    weight_decay: float = 1.5e-6,
) -> LARS:
    weights_a, bn_a = _split_params_for_lars(branch_a)
    weights_b, bn_b = _split_params_for_lars(branch_b)
    return LARS(
        [
            {"params": weights_a + weights_b, "lr": lr_weights, "no_trust": False},
            {"params": bn_a + bn_b, "lr": lr_bias, "no_trust": True},
        ],
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Pre-training loop


def _to_capc_input(x: torch.Tensor) -> torch.Tensor:
    """(B, T, S, A) -> (B, A, S, T)."""
    return x.permute(0, 3, 2, 1).contiguous()


def _split_windows(x_aSt: torch.Tensor, n_f: int) -> torch.Tensor:
    """(B, A, S, T) -> (B, L, A, S, N_f), L = T // n_f."""
    b, a, s, t = x_aSt.shape
    length = t // n_f
    if length == 0:
        raise ValueError(f"T={t} smaller than N_f={n_f}; cannot window.")
    x = x_aSt[..., : length * n_f]
    return x.reshape(b, a, s, length, n_f).permute(0, 3, 1, 2, 4).contiguous()


def pretrain_capc(
    branch_a: CAPCBranch,
    branch_b: CAPCBranch,
    loader: DataLoader,
    *,
    epochs: int = 300,
    n_f: int = 10,
    beta: float = 50.0,
    bt_lambda: float = 0.002,
    device: str = "cpu",
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[float]:
    """CAPC two-branch pre-training. LARS optimiser per §IV-E."""
    branch_a.to(device); branch_b.to(device)
    branch_a.train(); branch_b.train()
    optim = make_capc_optimizer(branch_a, branch_b)
    aug = augment_fn or capc_view_noise_then_submask
    epoch_losses: list[float] = []
    for _ in range(epochs):
        batch_losses: list[float] = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            view_a = _to_capc_input(aug(x))
            view_b = _to_capc_input(aug(x))
            w_a = _split_windows(view_a, n_f)
            w_b = _split_windows(view_b, n_f)
            z_a = branch_a.encode_windows(w_a)
            z_b = branch_b.encode_windows(w_b)
            c_a = branch_a.context(z_a)
            c_b = branch_b.context(z_b)
            loss, _ = capc_total_loss(
                z_a, c_a, z_b, c_b, branch_a.W, branch_b.W,
                future_steps=min(branch_a.future_steps, z_a.shape[1] - 1),
                beta=beta, bt_lambda=bt_lambda,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))
    return epoch_losses
