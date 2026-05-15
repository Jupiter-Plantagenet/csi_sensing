"""Project-baseline methods (supervised, SimCLR-trivial, SimCLR-handcrafted) on
Widar3.0 BVP.

After Gate 1 raw-CSI sanity failed at receivers=[1], [1,2,3], and [1..6]
(see results/2026-05-15-cross-subject-floor-finding.md and the
``results/2026-05-15-josiah-supervised-seed42`` aggregate), the project
baselines pivot to BVP — the published representation that the AutoFi paper
and the SenseFi benchmark both use.

This module mirrors ``ssl.py`` / ``eval.py`` / ``encoder.py`` for the BVP
input shape ``(B, 22, 20, 20)``. The augmentation set for ``simclr-trivial``
and ``simclr-handcrafted`` is the BVP analog of Slice 1's:

* ``random_temporal_crop`` — random window along the 22-step time axis,
  zero-padded back. The trivial-aug baseline.
* ``bvp_gaussian_then_temporal_mask`` — Gaussian noise + a random contiguous
  time-axis patch zeroed out. The hand-crafted-aug comparison column.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
import numpy as np

from .widar_bvp import BVP_T, BVP_VX, BVP_VY

NUM_PROJECT_CLASSES = 6  # canonical project filter: gestures 1-6


# -----------------------------------------------------------------------------
# Encoder shared across the three project baselines on BVP.


class BVPEncoder(nn.Module):
    """3-layer 2D CNN encoder for BVP ``(B, 22, 20, 20)`` -> ``(B, feat_dim)``.

    Treats the 22 time-steps as input channels; spatial convs over the
    ``(20, 20)`` velocity grid. Parameter count is in the same order as the
    raw-CSI TinyCNN so encoder capacity is not the comparison knob.
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(BVP_T, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BVPSupervisedClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_PROJECT_CLASSES, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = BVPEncoder(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# -----------------------------------------------------------------------------
# Augmentations.


def random_temporal_crop_bvp(x: torch.Tensor, crop_ratio: float = 0.7) -> torch.Tensor:
    """Random temporal window along axis ``-3`` (= 22 time-steps for BVP)."""
    t = x.shape[-3]
    t_crop = max(1, int(t * crop_ratio))
    max_start = t - t_crop
    start = int(torch.randint(0, max_start + 1, (1,)).item())
    out = torch.zeros_like(x)
    if x.ndim == 3:
        out[:t_crop] = x[start : start + t_crop]
    else:
        out[:, :t_crop] = x[:, start : start + t_crop]
    return out


def bvp_gaussian_then_temporal_mask(
    x: torch.Tensor, sigma: float = 0.05, mask_ratio: float = 0.15
) -> torch.Tensor:
    """Gaussian noise then a contiguous time-axis patch zeroed."""
    y = x + torch.randn_like(x) * sigma
    t = y.shape[-3]
    m_len = max(1, int(t * mask_ratio))
    if x.ndim == 3:
        start = int(torch.randint(0, t - m_len + 1, (1,)).item())
        y[start : start + m_len] = 0.0
    else:
        b = y.shape[0]
        for i in range(b):
            start = int(torch.randint(0, t - m_len + 1, (1,)).item())
            y[i, start : start + m_len] = 0.0
    return y


# -----------------------------------------------------------------------------
# SimCLR machinery.


class BVPSimCLR(nn.Module):
    def __init__(self, encoder: BVPEncoder, projection_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.feature_dim
        self.projection = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * b, device=z.device)
    targets = (targets + b) % (2 * b)
    return F.cross_entropy(sim, targets)


def pretrain_simclr_bvp(
    model: BVPSimCLR,
    loader: DataLoader,
    *,
    epochs: int,
    augment_fn,
    lr: float = 1e-3,
    temperature: float = 0.5,
    device: str = "cpu",
    log_every: int = 20,
) -> list[float]:
    """GPU-friendly SimCLR pre-training loop.

    Key optimisations over a naive loop on the small BVP CNN:

    * Loss accumulated as a GPU tensor; ``.item()`` only at epoch
      boundary so the per-batch CPU<->GPU sync goes away.
    * Augmentations are batched (no Python per-sample loops in
      ``doppler_warp`` / ``coherent_block_mask`` / ``static_perturb``).
    """
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        total = torch.zeros((), device=device)
        n = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).float()
            v1 = augment_fn(x)
            v2 = augment_fn(x)
            z1 = model(v1)
            z2 = model(v2)
            loss = _nt_xent(z1, z2, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total = total + loss.detach()
            n += 1
        avg = float(total.item()) / max(1, n)
        history.append(avg)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[bvp-simclr] epoch {epoch+1}/{epochs} nt-xent={avg:.4f}")
    return history


# -----------------------------------------------------------------------------
# Supervised training + evaluation.


def train_supervised_bvp(
    model: BVPSupervisedClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: list[float] = []
    for _ in range(epochs):
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        history.append(total / max(1, n))
    return history


@torch.no_grad()
def evaluate_bvp(
    model: BVPSupervisedClassifier,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        preds = model(x).argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def _extract_bvp_features(
    encoder: BVPEncoder, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval().to(device)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device).float()
        h = encoder(x)
        feats.append(h.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def linear_probe_bvp(
    encoder: BVPEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str = "cpu",
    max_iter: int = 1000,
    seed: int = 42,
) -> float:
    tr_x, tr_y = _extract_bvp_features(encoder, train_loader, device)
    te_x, te_y = _extract_bvp_features(encoder, test_loader, device)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed).fit(tr_x, tr_y)
    return float(np.mean(clf.predict(te_x) == te_y))


# -----------------------------------------------------------------------------
# Runner entry point used by production_runner.


def run_bvp_project_method(
    mode: str,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    bvp_root: str = "data/widar3/Widardata",
    cache_dir: str = "data/widar3/cache",
    gestures: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    num_classes: int = NUM_PROJECT_CLASSES,
    feature_dim: int = 128,
) -> float:
    """Single-seed BVP project-baseline run. Returns top-1 / linear-probe acc."""
    import random
    from .widar_bvp import WidarBVP

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gestures_list = list(gestures)
    cache_tag = (
        f"josiah-bvp-cs-g{'-'.join(map(str, gestures_list))}"
    )
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
    print(
        f"[josiah-bvp] cross-subject; train={len(train_ds)}, test={len(test_ds)}, "
        f"gestures={gestures_list}, classes={num_classes}"
    )

    use_cuda = torch.cuda.is_available()
    nworkers = 4 if use_cuda else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(mode != "supervised"),
        num_workers=nworkers,
        persistent_workers=(nworkers > 0),
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        persistent_workers=(nworkers > 0),
        pin_memory=use_cuda,
    )

    if mode == "supervised":
        model = BVPSupervisedClassifier(num_classes=num_classes, feature_dim=feature_dim)
        history = train_supervised_bvp(model, train_loader, epochs=epochs, device=device)
        print(f"[josiah-bvp-supervised] losses: {[round(l, 3) for l in history]}")
        acc = evaluate_bvp(model, test_loader, device=device)
        print(f"[josiah-bvp-supervised] top-1: {acc:.4f}")
        return acc

    proposed_modes = (
        "simclr-doppler",            # Slice 1
        "simclr-static-perturb",     # Slice 2
        "simclr-velocity-jitter",    # Slice 3 (BVP-reframed)
        "simclr-coherent-mask",      # Slice 4
        "simclr-doppler-coherent",   # Slice 6
    )
    if mode in ("simclr-trivial", "simclr-handcrafted") or mode in proposed_modes:
        if mode == "simclr-trivial":
            augment_fn = random_temporal_crop_bvp
        elif mode == "simclr-handcrafted":
            augment_fn = bvp_gaussian_then_temporal_mask
        elif mode == "simclr-doppler":
            # Slice 1 (George): Doppler-aware time warping. Shape-agnostic
            # on (T, X, Y); on BVP it scales gesture speed in velocity space.
            from src.slices.george.augmentations import doppler_warp
            augment_fn = doppler_warp
        elif mode == "simclr-static-perturb":
            # Slice 2 (Chigozie): time-mean velocity-profile swap across batch.
            # static_dynamic_split's time-mean variant works on any (B, T, X, Y).
            from src.slices.chigozie.augmentations import static_perturb
            augment_fn = static_perturb
        elif mode == "simclr-coherent-mask":
            # Slice 4 (Ihunanya): contiguous block mask on axis -2.
            # On BVP, axis -2 is vx -> "coherent velocity-band mask".
            from src.slices.ihunanya.augmentations import coherent_block_mask

            def augment_fn(x: torch.Tensor) -> torch.Tensor:
                # BVP vx has only 20 cells; a 5-wide block matches the
                # 25% mask ratio of the SimCLR-handcrafted baseline.
                return coherent_block_mask(x, block_width=5)

        elif mode == "simclr-doppler-coherent":
            # Slice 6 (Victor): composition of Doppler-warp and coherent-mask.
            from src.slices.victor.augmentations import doppler_then_coherent_mask

            def augment_fn(x: torch.Tensor) -> torch.Tensor:
                return doppler_then_coherent_mask(x, block_width=5)

        elif mode == "simclr-velocity-jitter":
            # Slice 3 (Collins, BVP-reframed): random affine in (vx, vy) plane.
            from src.slices.collins.bvp_velocity_jitter import bvp_velocity_jitter
            augment_fn = bvp_velocity_jitter
        else:
            raise ValueError(f"unhandled mode: {mode!r}")
        encoder = BVPEncoder(feature_dim=feature_dim)
        ssl_model = BVPSimCLR(encoder=encoder, projection_dim=64)
        history = pretrain_simclr_bvp(
            ssl_model,
            train_loader,
            epochs=epochs,
            augment_fn=augment_fn,
            device=device,
        )
        print(f"[josiah-bvp-{mode}] ssl losses tail: {[round(l, 3) for l in history[-5:]]}")
        probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        acc = linear_probe_bvp(
            ssl_model.encoder, probe_train_loader, test_loader, device=device, seed=seed
        )
        print(f"[josiah-bvp-{mode}] linear-probe: {acc:.4f}")
        return acc

    raise ValueError(f"unknown bvp mode: {mode!r}")
