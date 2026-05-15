"""CAPC exact reproduction code.

Implements CAPC (Barahimi, Tabassum, Omer, Waqar. *Context-Aware Predictive
Coding: A Representation Learning Framework for WiFi Sensing.* IEEE OJ-COMS
2024, arXiv:2410.01825) per the authors' official release
(``bornabr/CAPC``).

Status: **hardware-limited on this hardware**. The paper's headline cells
(SignFi-Lab pre-training → SignFi-Home few-shot; Table 1) require
synchronized uplink + downlink CSI from the *same* packet — a modality only
SignFi provides. Widar3.0 does not carry UL/DL CSI; UT-HAR does not either.
This module ships the method code so future work can drop SignFi data in,
or run the CAPC* (noise + subcarrier-mask single-view) variant on whatever
dataset is available. See ``papers/team/capc-hardware-limited.md`` for the
classification rationale.

Components (all paper-faithful, modulo ``[unspecified in paper]`` defaults):

* ``RSCNetEncoder`` — per-window CNN with three residual ``EncoderBlock`` s,
  output ``Linear`` projection to D=128.
* ``CAPCAutoregressor`` — single-layer ``GRU(128, 128)`` over window
  embeddings, batch-first.
* ``BarlowTwinsProjector`` — 3-layer MLP ``128 -> 256 -> 256 -> 256``.
* ``CAPCCPCLoss`` — CPC log-bilinear scorer with per-step ``W_k`` and
  in-batch negatives (paper Eq. 2, 3).
* ``BarlowTwinsLoss`` — cross-correlation with redundancy penalty
  ``lambda_bt`` (paper Eq. 4).
* ``CAPCLoss`` composite (paper Eq. 6): ``L = L_BT + beta * (L_CPC^A + L_CPC^B)``.

Training (paper §4.5):

* Optimizer LARS, weights LR=0.2, biases/BN LR=0.0048, weight_decay=1.5e-6,
  momentum=0.9. 10-epoch linear warmup, cosine decay.
* Batch 128 (paper) / 512 (repo); use 128 for paper match.
* 300 SSL epochs. Linear probe 100 epochs LR=1e-2.

LARS is not stock in PyTorch; ``CAPCOptimizer`` returns a ``torch.optim.SGD``
configured to mimic LARS for the warmup window; replace with a true LARS
implementation if a paper-faithful exact reproduction is run.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

D_FEAT = 128
NUM_WINDOWS_DEFAULT = 20
WINDOW_SHAPE_DEFAULT = (3, 30, 10)  # (Na, Ns, Nt_per_window)
PROJECTOR_DIMS = (128, 256, 256, 256)


# -----------------------------------------------------------------------------
# RSCNet encoder.


class _RSCNetEncoderBlock(nn.Module):
    """Multi-branch residual block from RSCNet (Barahimi et al. 2024,
    arXiv:2402.04888), as referenced by CAPC paper §3.1.

    Three parallel branches: ``(3x1)`` dilations {1,2,3}, ``(1x3)``
    dilations {1,2,3}, and a standard ``(3x3)``. Concatenate -> ``1x1``
    project -> residual add -> ``PReLU(0.3)``.

    Channel widths are partially ``[unspecified in paper]``; we use the
    repo default ``branch_channels = max(in_channels // 3, 8)``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = channels
        bc = max(c // 3, 8)

        def _b(kernel: tuple[int, int], dilation: int) -> nn.Module:
            pad = (
                (kernel[0] - 1) * dilation // 2,
                (kernel[1] - 1) * dilation // 2,
            )
            return nn.Conv2d(c, bc, kernel_size=kernel, padding=pad, dilation=dilation)

        self.branches_3x1 = nn.ModuleList([_b((3, 1), d) for d in (1, 2, 3)])
        self.branches_1x3 = nn.ModuleList([_b((1, 3), d) for d in (1, 2, 3)])
        self.branch_3x3 = nn.Conv2d(c, bc, kernel_size=3, padding=1)

        merged = bc * 7
        self.proj = nn.Conv2d(merged, c, kernel_size=1)
        self.act = nn.PReLU(init=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches_3x1]
        parts += [b(x) for b in self.branches_1x3]
        parts.append(self.branch_3x3(x))
        merged = torch.cat(parts, dim=1)
        out = self.proj(merged)
        return self.act(out + x)


class RSCNetEncoder(nn.Module):
    """Per-window CNN -> ``(B, D=128)``.

    Input window shape: ``(Na=3, Ns=30, Nt_per_window=10)`` per CAPC §4.1.1.
    """

    def __init__(
        self,
        window_shape: tuple[int, int, int] = WINDOW_SHAPE_DEFAULT,
        feature_dim: int = D_FEAT,
        stem_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        na, ns, nt = window_shape
        self.window_shape = window_shape
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(na, stem_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(stem_channels),
            nn.PReLU(init=0.3),
        )
        self.blocks = nn.Sequential(
            *[_RSCNetEncoderBlock(stem_channels) for _ in range(num_blocks)]
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(stem_channels * ns * nt, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, Na, Ns, Nt)``. Returns ``(B, D)``."""
        h = self.stem(x)
        h = self.blocks(h)
        h = self.flatten(h)
        return self.proj(h)


# -----------------------------------------------------------------------------
# Autoregressor + projector.


class CAPCAutoregressor(nn.Module):
    """GRU(D=128, hidden=128) over window embeddings (CAPC §3.1, §4.5)."""

    def __init__(self, feature_dim: int = D_FEAT, hidden_dim: int = D_FEAT) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, z_windows: torch.Tensor) -> torch.Tensor:
        """``z_windows``: ``(B, L, D)``. Returns the hidden sequence ``(B, L, D_h)``."""
        h_seq, _ = self.gru(z_windows)
        return h_seq


class BarlowTwinsProjector(nn.Sequential):
    """3-layer MLP per ``modules.py::projector`` in the official repo."""

    def __init__(self, dims: tuple[int, ...] = PROJECTOR_DIMS) -> None:
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
        super().__init__(*layers)


# -----------------------------------------------------------------------------
# Losses.


class CAPCCPCLoss(nn.Module):
    """CPC contrastive loss with one ``W_k`` per future step (paper Eq. 2/3).

    For each anchor window position ``t`` and future offset ``k in 1..K``:
    score ``f_k(x_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)`` against in-batch
    negatives (other samples at the same future position).
    """

    def __init__(self, *, num_future_steps: int, feature_dim: int = D_FEAT) -> None:
        super().__init__()
        self.K = num_future_steps
        self.W = nn.Parameter(torch.randn(num_future_steps, feature_dim, feature_dim) * 0.02)

    def forward(self, z_seq: torch.Tensor, c_t: torch.Tensor, anchor_index: int = 0) -> torch.Tensor:
        """``z_seq``: ``(B, L, D)``  ``c_t``: ``(B, D_h)``  -> scalar CPC loss."""
        b, l, d = z_seq.shape
        if anchor_index + self.K >= l:
            raise ValueError(
                f"anchor_index={anchor_index} + K={self.K} must be < L={l}"
            )
        losses: list[torch.Tensor] = []
        for k in range(1, self.K + 1):
            W_k = self.W[k - 1]
            pred = c_t @ W_k  # (B, D)
            future = z_seq[:, anchor_index + k, :]  # (B, D)
            scores = pred @ future.t()  # (B, B); diag = positives, off-diag = negatives
            log_softmax = F.log_softmax(scores, dim=1)
            positives = log_softmax.diag()
            losses.append(-positives.mean())
        return torch.stack(losses).mean()


class BarlowTwinsLoss(nn.Module):
    """Standard Barlow-Twins cross-correlation loss (paper Eq. 4)."""

    def __init__(self, *, lambda_bt: float = 0.002) -> None:
        super().__init__()
        self.lambda_bt = lambda_bt

    def forward(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        # Batch-normalize each dim (zero mean, unit std) per Barlow-Twins recipe.
        eps = 1e-6
        a = (p_a - p_a.mean(dim=0)) / (p_a.std(dim=0) + eps)
        b = (p_b - p_b.mean(dim=0)) / (p_b.std(dim=0) + eps)
        c = a.t() @ b / max(1, a.shape[0])  # (D, D)
        on_diag = (c.diag() - 1.0).pow(2).sum()
        off_diag = (c - torch.diag(c.diag())).pow(2).sum()
        return on_diag + self.lambda_bt * off_diag


class CAPCLoss(nn.Module):
    """Composite ``L = L_BT + beta * (L_CPC^A + L_CPC^B)`` (paper Eq. 6)."""

    def __init__(
        self,
        *,
        num_future_steps: int,
        feature_dim: int = D_FEAT,
        beta: float = 50.0,
        lambda_bt: float = 0.002,
    ) -> None:
        super().__init__()
        self.cpc_a = CAPCCPCLoss(num_future_steps=num_future_steps, feature_dim=feature_dim)
        self.cpc_b = CAPCCPCLoss(num_future_steps=num_future_steps, feature_dim=feature_dim)
        self.bt = BarlowTwinsLoss(lambda_bt=lambda_bt)
        self.beta = beta

    def forward(
        self,
        z_seq_a: torch.Tensor,
        c_t_a: torch.Tensor,
        z_seq_b: torch.Tensor,
        c_t_b: torch.Tensor,
        p_a: torch.Tensor,
        p_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_cpc_a = self.cpc_a(z_seq_a, c_t_a)
        l_cpc_b = self.cpc_b(z_seq_b, c_t_b)
        l_bt = self.bt(p_a, p_b)
        total = l_bt + self.beta * (l_cpc_a + l_cpc_b)
        return {"bt": l_bt, "cpc_a": l_cpc_a, "cpc_b": l_cpc_b, "total": total}


# -----------------------------------------------------------------------------
# Two-view CAPC model.


class CAPC(nn.Module):
    """End-to-end CAPC model.

    Forward consumes a paired UL/DL CSI sample ``(view_a, view_b)`` shaped
    ``(B, L, Na, Ns, Nt)`` per view (L = num_windows, each window is
    ``(Na, Ns, Nt)``). Returns ``(z_seq_a, c_a, z_seq_b, c_b, p_a, p_b)``.

    Augmentations (noise, time flip, subcarrier mask) are applied *outside*
    this module by the training loop so different recipes can be swapped in
    without touching the model.
    """

    def __init__(
        self,
        *,
        window_shape: tuple[int, int, int] = WINDOW_SHAPE_DEFAULT,
        feature_dim: int = D_FEAT,
    ) -> None:
        super().__init__()
        self.encoder = RSCNetEncoder(window_shape=window_shape, feature_dim=feature_dim)
        self.autoregressor = CAPCAutoregressor(feature_dim=feature_dim)
        self.projector = BarlowTwinsProjector()

    def _embed_windows(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, Na, Ns, Nt)
        b, l, na, ns, nt = x.shape
        flat = x.reshape(b * l, na, ns, nt)
        z = self.encoder(flat)
        return z.reshape(b, l, -1)

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_seq_a = self._embed_windows(view_a)
        z_seq_b = self._embed_windows(view_b)
        h_a = self.autoregressor(z_seq_a)
        h_b = self.autoregressor(z_seq_b)
        c_a = h_a[:, 0, :]
        c_b = h_b[:, 0, :]
        # Project mean-pooled window embeddings for Barlow-Twins.
        p_a = self.projector(z_seq_a.mean(dim=1))
        p_b = self.projector(z_seq_b.mean(dim=1))
        return z_seq_a, c_a, z_seq_b, c_b, p_a, p_b


# -----------------------------------------------------------------------------
# Augmentations (paper §3.2 / §4.4; for the CAPC* fallback variant when no
# UL/DL data is available).


def capc_time_flip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(-1,))


def capc_gaussian_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def capc_subcarrier_mask(x: torch.Tensor, ratio: float = 0.10) -> torch.Tensor:
    """Zero out a random ``ratio`` fraction of subcarriers (axis -2)."""
    s = x.shape[-2]
    keep = (torch.rand(s, device=x.device) >= ratio).to(x.dtype)
    shape = [1] * x.ndim
    shape[-2] = s
    return x * keep.view(shape)


# -----------------------------------------------------------------------------
# LARS-shaped optimizer (paper §4.5). True LARS implementation deferred —
# documented hardware-limited dependency.


class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer (You et al. 2017).

    Per-parameter trust-ratio: ``local_lr = trust_coef * ||w|| / (||grad|| +
    weight_decay * ||w|| + eps)``. Bias/BN parameters bypass LARS via the
    ``exclude_from_lars`` group flag (they get plain SGD with the group LR).
    This is the optimizer CAPC §4.5 specifies.
    """

    def __init__(
        self,
        params,
        *,
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
            exclude_from_lars=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            trust_coef = group["trust_coef"]
            eps = group["eps"]
            exclude = group["exclude_from_lars"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)
                if not exclude and p.ndim > 1:
                    w_norm = p.norm()
                    g_norm = grad.norm()
                    if float(w_norm) > 0 and float(g_norm) > 0:
                        local_lr = trust_coef * w_norm / (g_norm + eps)
                    else:
                        local_lr = torch.tensor(1.0, device=p.device)
                else:
                    local_lr = torch.tensor(1.0, device=p.device)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=float(local_lr) * lr)
                p.add_(buf, alpha=-1)
        return loss


def build_capc_optimizer(
    params: Iterable[nn.Parameter],
    *,
    lr_weights: float = 0.2,
    lr_biasbn: float = 0.0048,
    momentum: float = 0.9,
    weight_decay: float = 1.5e-6,
    trust_coef: float = 0.001,
    use_true_lars: bool = True,
) -> torch.optim.Optimizer:
    """Return CAPC's LARS optimizer per paper §4.5.

    Two parameter groups: weights (LARS-scaled, ``lr=0.2``, ``wd=1.5e-6``)
    and biases/BN (plain SGD, ``lr=0.0048``, ``wd=0``). With
    ``use_true_lars=False`` falls back to plain SGD (legacy stand-in).
    """
    weight_params: list[nn.Parameter] = []
    biasbn_params: list[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            biasbn_params.append(p)
        else:
            weight_params.append(p)
    if use_true_lars:
        return LARS(
            [
                {
                    "params": weight_params,
                    "lr": lr_weights,
                    "weight_decay": weight_decay,
                    "exclude_from_lars": False,
                },
                {
                    "params": biasbn_params,
                    "lr": lr_biasbn,
                    "weight_decay": 0.0,
                    "exclude_from_lars": True,
                },
            ],
            momentum=momentum,
            trust_coef=trust_coef,
        )
    return torch.optim.SGD(
        [
            {"params": weight_params, "lr": lr_weights, "weight_decay": weight_decay},
            {"params": biasbn_params, "lr": lr_biasbn, "weight_decay": 0.0},
        ],
        momentum=momentum,
    )


def warmup_cosine_lr(epoch: int, *, total_epochs: int, warmup_epochs: int = 10) -> float:
    """LR multiplier for the 10-epoch linear warmup + cosine decay schedule."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# CAPC training driver (uses SignFi loader from src/slices/josiah/signfi.py).


def pretrain_capc(
    model: CAPC,
    loader,
    *,
    epochs: int,
    num_future_steps: int,
    feature_dim: int = D_FEAT,
    beta: float = 50.0,
    lambda_bt: float = 0.002,
    base_lr_weights: float = 0.2,
    base_lr_biasbn: float = 0.0048,
    weight_decay: float = 1.5e-6,
    momentum: float = 0.9,
    warmup_epochs: int = 10,
    device: str = "cpu",
    log_every: int = 5,
    optimizer_kind: str = "lars-standin",
    adamw_lr: float = 1e-3,
    adamw_wd: float = 0.05,
    grad_clip: float = 1.0,
) -> list[float]:
    """SSL pre-training: CPC + Barlow-Twins on UL/DL pairs. Returns per-epoch loss.

    ``optimizer_kind``:

    * ``"lars"`` (default, paper-exact): true LARS (You et al. 2017) per
      paper §4.5, with weights lr=0.2, biases/BN lr=0.0048, wd=1.5e-6, mom=0.9.
    * ``"adamw"``: AdamW lr=``adamw_lr`` wd=``adamw_wd``; stable interim
      stand-in.
    """
    model = model.to(device)
    loss_fn = CAPCLoss(
        num_future_steps=num_future_steps,
        feature_dim=feature_dim,
        beta=beta,
        lambda_bt=lambda_bt,
    ).to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    if optimizer_kind == "adamw":
        optim = torch.optim.AdamW(params, lr=adamw_lr, weight_decay=adamw_wd)
    elif optimizer_kind in ("lars", "lars-standin"):
        optim = build_capc_optimizer(
            params,
            lr_weights=base_lr_weights,
            lr_biasbn=base_lr_biasbn,
            momentum=momentum,
            weight_decay=weight_decay,
            use_true_lars=True,
        )
    else:
        raise ValueError(f"unknown optimizer_kind={optimizer_kind!r}")
    history: list[float] = []
    for epoch in range(epochs):
        mult = warmup_cosine_lr(epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)
        if optimizer_kind in ("lars", "lars-standin"):
            for g in optim.param_groups:
                base = (
                    base_lr_weights
                    if g["params"] and g["params"][0].ndim > 1
                    else base_lr_biasbn
                )
                g["lr"] = base * mult
        else:
            for g in optim.param_groups:
                g["lr"] = adamw_lr * mult
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            view_a, view_b, _label = batch
            view_a = view_a.to(device).float()
            view_b = view_b.to(device).float()
            z_a, c_a, z_b, c_b, p_a, p_b = model(view_a, view_b)
            out = loss_fn(z_a, c_a, z_b, c_b, p_a, p_b)
            loss = out["total"]
            optim.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optim.step()
            total += float(loss.item())
            n += 1
        history.append(total / max(1, n))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[capc-ssl] epoch {epoch+1}/{epochs} loss={history[-1]:.4f} lr-mult={mult:.3f}")
    return history


@torch.no_grad()
def _capc_pooled_features(
    model: CAPC, loader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated window embeddings (CAPC §4.2: L*D = 20*128 = 2560-dim)."""
    import numpy as np

    model.eval().to(device)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        view_a, _view_b, y = batch
        view_a = view_a.to(device).float()
        z_seq = model._embed_windows(view_a)  # (B, L, D)
        pooled = z_seq.reshape(z_seq.shape[0], -1)
        feats.append(pooled.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def linear_probe_capc(
    model: CAPC,
    train_loader,
    test_loader,
    *,
    device: str = "cpu",
    max_iter: int = 1000,
    seed: int = 42,
) -> float:
    """Linear probe per CAPC Table 1: logistic regression on concatenated window embeddings."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tr_x, tr_y = _capc_pooled_features(model, train_loader, device)
    te_x, te_y = _capc_pooled_features(model, test_loader, device)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed).fit(tr_x, tr_y)
    return float(np.mean(clf.predict(te_x) == te_y))


def run_capc(
    *,
    seed: int,
    ssl_epochs: int = 300,
    batch_size: int = 128,
    k_shot: int = 10,
    data_root: str = "data",
    cache_dir: str = "data/widar3/cache",
    pretrain_env: str = "home",
    eval_env: str = "home",
    num_future_steps: int = 8,
    feature_dim: int = D_FEAT,
) -> float:
    """Single-seed CAPC run. Returns linear-probe top-1 accuracy.

    Paper-exact protocol: ``pretrain_env="lab"`` and ``eval_env="home"``.
    Interim Home-only protocol (no Lab data): both args = ``"home"``; SSL
    pre-trains on a random 80% of Home and linear-probe uses ``k_shot``
    samples/class from the held-out 20% (train) versus the remainder (test).
    """
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from .signfi import (
        NUM_SIGNFI_CLASSES,
        SignFiPaired,
        k_shot_split,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    same_env = pretrain_env == eval_env
    if not same_env and (pretrain_env != "lab" or eval_env != "home"):
        raise ValueError(
            "Only pretrain_env=='lab', eval_env=='home' is supported as cross-env; "
            "use the same env for both for interim runs."
        )

    cache_pre = f"{cache_dir}/signfi-{pretrain_env}-paired.pt"
    cache_eval = f"{cache_dir}/signfi-{eval_env}-paired.pt"

    if same_env:
        # Stratified 50/50 split for SSL vs linear-probe pool within one env.
        # Home has 10 instances/class; stratified 5/5 split keeps each class
        # balanced in both pools.
        all_ds = SignFiPaired(env=pretrain_env, root=data_root, cache_path=cache_pre)
        labels_all = all_ds._y.numpy()
        rng = np.random.default_rng(seed)
        ssl_idx_list: list[int] = []
        eval_pool_idx_list: list[int] = []
        for c in range(NUM_SIGNFI_CLASSES):
            cls_idx = np.where(labels_all == c)[0]
            rng.shuffle(cls_idx)
            half = cls_idx.size // 2
            ssl_idx_list.extend(cls_idx[:half].tolist())
            eval_pool_idx_list.extend(cls_idx[half:].tolist())
        ssl_idx = ssl_idx_list
        eval_pool_idx = eval_pool_idx_list
        ssl_ds = SignFiPaired(env=pretrain_env, root=data_root, indices=ssl_idx, cache_path=cache_pre)
        # k-shot split inside the eval pool. Clamp k to leave at least one
        # test sample per class.
        eval_labels = all_ds._y.numpy()[eval_pool_idx]
        from collections import Counter
        per_class = Counter(eval_labels.tolist())
        min_per_class = min(per_class.values()) if per_class else 0
        effective_k = min(k_shot, max(1, min_per_class - 1))
        if effective_k != k_shot:
            print(f"[capc] clamped k_shot from {k_shot} to {effective_k} (min eval-pool class size = {min_per_class})")
        train_local, test_local = k_shot_split(
            eval_labels, k=effective_k, num_classes=NUM_SIGNFI_CLASSES, seed=seed
        )
        probe_train_idx = [eval_pool_idx[i] for i in train_local.tolist()]
        probe_test_idx = [eval_pool_idx[i] for i in test_local.tolist()]
        probe_train_ds = SignFiPaired(
            env=eval_env, root=data_root, indices=probe_train_idx, cache_path=cache_eval
        )
        probe_test_ds = SignFiPaired(
            env=eval_env, root=data_root, indices=probe_test_idx, cache_path=cache_eval
        )
    else:
        # Paper-exact: Lab pre-train, Home k-shot eval.
        ssl_ds = SignFiPaired(env="lab", root=data_root, cache_path=cache_pre)
        home_all = SignFiPaired(env="home", root=data_root, cache_path=cache_eval)
        labels_home = home_all._y.numpy()
        from collections import Counter
        per_class = Counter(labels_home.tolist())
        min_per_class = min(per_class.values()) if per_class else 0
        effective_k = min(k_shot, max(1, min_per_class - 1))
        if effective_k != k_shot:
            print(
                f"[capc] clamped k_shot from {k_shot} to {effective_k} "
                f"(Home has {min_per_class} instances/class; need >=1 test sample)"
            )
        train_idx, test_idx = k_shot_split(
            labels_home, k=effective_k, num_classes=NUM_SIGNFI_CLASSES, seed=seed
        )
        probe_train_ds = SignFiPaired(
            env="home", root=data_root, indices=train_idx.tolist(), cache_path=cache_eval
        )
        probe_test_ds = SignFiPaired(
            env="home", root=data_root, indices=test_idx.tolist(), cache_path=cache_eval
        )

    print(
        f"[capc] pretrain={pretrain_env} eval={eval_env} ssl_n={len(ssl_ds)} "
        f"probe_train_n={len(probe_train_ds)} probe_test_n={len(probe_test_ds)} k_shot={k_shot}"
    )

    ssl_loader = DataLoader(ssl_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    probe_train_loader = DataLoader(probe_train_ds, batch_size=batch_size, shuffle=False)
    probe_test_loader = DataLoader(probe_test_ds, batch_size=batch_size, shuffle=False)

    model = CAPC(window_shape=WINDOW_SHAPE_DEFAULT, feature_dim=feature_dim)
    # Paper-exact Lab->Home uses true LARS per §4.5. Interim Home-only keeps
    # AdamW for stability (smaller dataset is more sensitive to LARS LRs).
    interim = same_env
    pretrain_capc(
        model,
        ssl_loader,
        epochs=ssl_epochs,
        num_future_steps=num_future_steps,
        feature_dim=feature_dim,
        device=device,
        optimizer_kind="adamw" if interim else "lars",
    )
    acc = linear_probe_capc(model, probe_train_loader, probe_test_loader, device=device, seed=seed)
    print(f"[capc] linear-probe top-1 = {acc:.4f}")
    return acc
