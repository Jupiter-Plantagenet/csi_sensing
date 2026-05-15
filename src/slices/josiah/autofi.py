"""AutoFi exact reproduction on Widar3.0 BVP.

Implements AutoFi (Yang, Chen, Zou, Wang, Xie. *AutoFi: Toward Automatic Wi-Fi
Human Sensing via Geometric Self-Supervised Learning.* IoT-J 2022,
arXiv:2205.01629) per the authors' released SenseFi code
(``xyanchen/WiFi-CSI-Sensing-Benchmark``: ``self_supervised.py`` +
``self_supervised_model.py`` + ``dataset.py``).

Key fidelity choices:

* Two-stream encoder ``CNN_Parrallel`` adapted to Widar BVP input
  ``(22, 20, 20)`` (paper §IV-D says the first layer is modified to match
  the BVP shape; we keep the rest of SenseFi's three-conv structure).
* GSS loss = ``L_kl + (1+lam1) * EH - lam2 * HE + 100 * L_kde`` exactly as
  ``self_supervised.py::EntLoss::forward`` returns ``loss['final-kde']``.
* Augmentations: two views via additive Gaussian noise ``N(1, 2)`` scaled by
  ``epsilon ~ U(0, 2)`` and ``epsilon ~ U(0.1, 2)``.
* Optimizer: AdamW, lr=1e-3, weight_decay=1.5e-6 for SSL; Adam lr=1e-3,
  wd=1e-5 for the linear-probe classifier.
* Pre-training: 100 epochs. Linear probe: 300 epochs (matches the
  ``self_supervised.py`` schedule).

Reproduction targets and known gaps:

* SenseFi-style protocol (released code): SSL on all 22 classes, linear probe
  on the same data. This file targets that protocol exactly.
* Paper §IV-D headline (Widar BVP 20-shot 6-class FSC = 63.80%): NOT what
  the released code does. Paper used SGD + 300 SSL epochs + few-shot
  calibration with ``L_c + L_f``; reach via ``--protocol paper-fsc``.
* The BVP CSVs we have are the SenseFi-processed ``T=22`` format. The paper
  §IV-D uses the original Widar release with ``T=40``, so an exact match to
  63.80% is unreachable from the CSV release — classify hardware-limited.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .widar_bvp import BVP_T, BVP_VX, BVP_VY, NUM_BVP_CLASSES

# -----------------------------------------------------------------------------
# Encoder — adapted from SenseFi CNN_Parrallel for Widar BVP (22, 20, 20).


class AutoFiBVPEncoder(nn.Module):
    """SenseFi-style CNN encoder, first layer resized for BVP (22, 20, 20).

    Output shape ``(B, hidden_states)`` after BN, matching the SenseFi
    ``CNN_encoder`` projection-head output used as the SSL feature.
    """

    def __init__(self, hidden_states: int = 256) -> None:
        super().__init__()
        # Input: (B, 22, 20, 20). Treat the 22 time-steps as channels.
        self.encoder = nn.Sequential(
            nn.Conv2d(BVP_T, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Output spatial: 20 -> 20 -> 10 -> 5; channels 96 -> 96*5*5 = 2400.
        self.feat_dim = 96 * 5 * 5
        self.mapping = nn.Linear(self.feat_dim, hidden_states)
        self.bn = nn.BatchNorm1d(hidden_states)

    def forward(self, x: torch.Tensor, flag: str = "unsupervised") -> torch.Tensor:
        h = self.encoder(x)
        h = h.reshape(h.shape[0], -1)
        if flag == "supervised":
            return h
        return self.bn(self.mapping(h))


class AutoFiParallel(nn.Module):
    """Two-stream encoder + linear classifier, mirroring ``CNN_Parrallel``.

    During SSL both encoders see independently augmented views; during the
    supervised linear probe both encoders see the same ``x`` and the
    classifier reads the pre-projection ``(B, feat_dim)`` features.
    """

    def __init__(self, num_classes: int = NUM_BVP_CLASSES, hidden_states: int = 256) -> None:
        super().__init__()
        self.encoder_1 = AutoFiBVPEncoder(hidden_states=hidden_states)
        self.encoder_2 = AutoFiBVPEncoder(hidden_states=hidden_states)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_1.feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, flag: str = "unsupervised"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.encoder_1(x1, flag=flag)
        h2 = self.encoder_2(x2, flag=flag)
        if flag == "supervised":
            return self.classifier(h1), self.classifier(h2)
        return h1, h2


# -----------------------------------------------------------------------------
# GSS loss (EntLoss + cosine_similarity_loss) — verbatim from
# ``self_supervised.py``, vectorized as drop-in PyTorch.


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    return (p * (p + eps).log() - p * (q + eps).log()).sum(dim=1).mean()


def _eh(probs: torch.Tensor, eps: float) -> torch.Tensor:
    # Mean over batch of per-sample entropy.
    return -(probs * (probs + eps).log()).sum(dim=1).mean()


def _he(probs: torch.Tensor, eps: float) -> torch.Tensor:
    # Entropy of the batch-mean distribution.
    mean = probs.mean(dim=0)
    return -(mean * (mean + eps).log()).sum()


def _cosine_similarity_loss(out_net: torch.Tensor, tgt_net: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    out = F.normalize(out_net, dim=1, eps=eps)
    tgt = F.normalize(tgt_net, dim=1, eps=eps)
    s_out = (out @ out.t() + 1.0) / 2.0
    s_tgt = (tgt @ tgt.t() + 1.0) / 2.0
    s_out = s_out / s_out.sum(dim=1, keepdim=True)
    s_tgt = s_tgt / s_tgt.sum(dim=1, keepdim=True)
    return (s_tgt * ((s_tgt + eps) / (s_out + eps)).log()).mean()


class AutoFiGSSLoss(nn.Module):
    """SenseFi ``EntLoss::forward`` returning ``loss['final-kde']``.

    Returns a dict so callers can inspect components; ``loss['final-kde']`` is
    the scalar to ``.backward()``.
    """

    def __init__(
        self,
        *,
        tau: float = 1.0,
        eps: float = 1e-5,
        lam1: float = 0.0,
        lam2: float = 0.5,
        kde_weight: float = 100.0,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.lam1 = lam1
        self.lam2 = lam2
        self.kde_weight = kde_weight

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> dict[str, torch.Tensor]:
        probs1 = F.softmax(feat1, dim=-1)
        probs2 = F.softmax(feat2, dim=-1)
        sharp1 = F.softmax(feat1 / self.tau, dim=-1)
        sharp2 = F.softmax(feat2 / self.tau, dim=-1)

        kl = 0.5 * (_kl(probs1, probs2, self.eps) + _kl(probs2, probs1, self.eps))
        eh = 0.5 * (_eh(sharp1, self.eps) + _eh(sharp2, self.eps))
        he = 0.5 * (_he(sharp1, self.eps) + _he(sharp2, self.eps))
        kde = _cosine_similarity_loss(feat1, feat2, eps=self.eps)

        final = kl + ((1 + self.lam1) * eh - self.lam2 * he)
        final_kde = self.kde_weight * kde + final
        return {
            "kl": kl,
            "eh": eh,
            "he": he,
            "kde": kde,
            "final": final,
            "final-kde": final_kde,
        }


# -----------------------------------------------------------------------------
# Augmentation: gaussian_noise.


def gaussian_noise_bvp(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Additive Gaussian noise per ``self_supervised.py::gaussian_noise``.

    Reference noise is ``N(mean=1, std=2)`` over the input shape; ``epsilon``
    scales the noise. Sampled fresh per call so independent calls yield two
    independent views.
    """
    noise = torch.normal(mean=1.0, std=2.0, size=x.shape, device=x.device, dtype=x.dtype)
    return x + epsilon * noise


# -----------------------------------------------------------------------------
# Training loops.


def pretrain_autofi(
    model: AutoFiParallel,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1.5e-6,
    tau: float = 1.0,
    lam1: float = 0.0,
    lam2: float = 0.5,
    device: str = "cpu",
    eps_view1_range: tuple[float, float] = (0.0, 2.0),
    eps_view2_range: tuple[float, float] = (0.1, 2.0),
    log_every: int = 10,
) -> list[float]:
    """SSL pre-training. Returns per-epoch mean ``final-kde`` losses."""
    model = model.to(device)
    model.train()
    criterion = AutoFiGSSLoss(tau=tau, lam1=lam1, lam2=lam2)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    import random as _random

    history: list[float] = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            eps1 = _random.uniform(*eps_view1_range)
            eps2 = _random.uniform(*eps_view2_range)
            x1 = gaussian_noise_bvp(x, eps1)
            x2 = gaussian_noise_bvp(x, eps2)
            feat1, feat2 = model(x1, x2, flag="unsupervised")
            losses = criterion(feat1, feat2)
            loss = losses["final-kde"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
            n += 1
        avg = total / max(1, n)
        history.append(avg)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[autofi-ssl] epoch {epoch+1}/{epochs} final-kde={avg:.4f}")
    return history


def linear_probe_autofi(
    model: AutoFiParallel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    log_every: int = 50,
) -> dict[str, float]:
    """Freeze encoders; train ``model.classifier`` with cross-entropy.

    Matches ``self_supervised.py``'s ``Supervised classifier training``: only
    ``model.classifier.parameters()`` is optimized.
    """
    model = model.to(device)
    optim = torch.optim.Adam(
        model.classifier.parameters(), lr=lr, weight_decay=weight_decay
    )
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            y1, y2 = model(x, x, flag="supervised")
            loss = ce(y1, y) + ce(y2, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[autofi-probe] epoch {epoch+1}/{epochs} loss={total:.4f}")

    model.eval()
    correct_1 = 0
    correct_2 = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            y1, y2 = model(x, x, flag="supervised")
            correct_1 += int((y1.argmax(dim=1) == y).sum().item())
            correct_2 += int((y2.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return {
        "accuracy_branch1": correct_1 / max(1, total),
        "accuracy_branch2": correct_2 / max(1, total),
        "accuracy": max(correct_1, correct_2) / max(1, total),
    }


# -----------------------------------------------------------------------------
# Runner entry point used by production_runner.


def run_autofi(
    *,
    seed: int,
    ssl_epochs: int = 100,
    probe_epochs: int = 300,
    batch_size: int = 64,
    bvp_root: str = "data/widar3/Widardata",
    cache_dir: str = "data/widar3/cache",
    protocol: Literal["sensefi", "cross-subject"] = "sensefi",
    num_classes: int = NUM_BVP_CLASSES,
) -> float:
    """Single-seed AutoFi run on Widar BVP. Returns linear-probe accuracy.

    ``protocol``:

    * ``sensefi``: SSL pre-train on ``Widardata/train/`` (all 22 classes),
      linear probe trained on the same split, evaluated on
      ``Widardata/test/``. This is the protocol the AutoFi authors' released
      code implements.
    * ``cross-subject``: cross-subject split (train users 5-17 / test users
      1-4), all 22 classes. Closer to the project comparison protocol.
    """
    import random

    import numpy as np

    from .widar_bvp import WidarBVP

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_tag = f"josiah-autofi-{protocol}-bvp"
    train_ds = WidarBVP(
        root=bvp_root,
        split=protocol,
        train=True,
        cache_path=f"{cache_dir}/{cache_tag}-train.pt",
    )
    test_ds = WidarBVP(
        root=bvp_root,
        split=protocol,
        train=False,
        cache_path=f"{cache_dir}/{cache_tag}-test.pt",
    )
    print(
        f"[autofi] protocol={protocol}; train={len(train_ds)}, test={len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AutoFiParallel(num_classes=num_classes, hidden_states=256)
    pretrain_autofi(model, train_loader, epochs=ssl_epochs, device=device)
    metrics = linear_probe_autofi(
        model, probe_train_loader, test_loader, epochs=probe_epochs, device=device
    )
    print(
        f"[autofi] probe acc branch1={metrics['accuracy_branch1']:.4f} "
        f"branch2={metrics['accuracy_branch2']:.4f} max={metrics['accuracy']:.4f}"
    )
    return float(metrics["accuracy"])


# =============================================================================
# AutoFi UT-HAR — paper §IV-C exact reproduction target (20-shot = 0.788).


class AutoFiUTHAREncoder(nn.Module):
    """1D CNN encoder for UT-HAR ``(1, 250, 90)`` input.

    Adapted from SenseFi ``CNN_encoder``; UT-HAR has 90 features per
    timestep (30 subcarriers × 3 antennas), so 1D convolution along time
    with 90 input channels matches the input geometry better than the
    NTU-Fi-shaped 2D encoder. Three conv blocks preserve SenseFi's depth
    and the projection-head topology.
    """

    def __init__(self, hidden_states: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.feat_dim = 256
        self.mapping = nn.Linear(256, hidden_states)
        self.bn = nn.BatchNorm1d(hidden_states)

    def forward(self, x: torch.Tensor, flag: str = "unsupervised") -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1).permute(0, 2, 1)
        h = self.encoder(x)
        if flag == "supervised":
            return h
        return self.bn(self.mapping(h))


class AutoFiUTHARParallel(nn.Module):
    """Two-stream UT-HAR encoder + classifier head, mirroring AutoFiParallel."""

    def __init__(self, num_classes: int = 7, hidden_states: int = 256) -> None:
        super().__init__()
        self.encoder_1 = AutoFiUTHAREncoder(hidden_states=hidden_states)
        self.encoder_2 = AutoFiUTHAREncoder(hidden_states=hidden_states)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_1.feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, flag: str = "unsupervised"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.encoder_1(x1, flag=flag)
        h2 = self.encoder_2(x2, flag=flag)
        if flag == "supervised":
            return self.classifier(h1), self.classifier(h2)
        return h1, h2


def gaussian_noise_uthar(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """AutoFi gaussian noise per self_supervised.py::gaussian_noise."""
    noise = torch.normal(mean=1.0, std=2.0, size=x.shape, device=x.device, dtype=x.dtype)
    return x + epsilon * noise


def pretrain_autofi_uthar(
    model: AutoFiUTHARParallel,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1.5e-6,
    tau: float = 1.0,
    lam1: float = 0.0,
    lam2: float = 0.5,
    device: str = "cpu",
    eps_view1_range: tuple[float, float] = (0.0, 2.0),
    eps_view2_range: tuple[float, float] = (0.1, 2.0),
    log_every: int = 10,
) -> list[float]:
    import random as _random

    model = model.to(device)
    model.train()
    criterion = AutoFiGSSLoss(tau=tau, lam1=lam1, lam2=lam2)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            eps1 = _random.uniform(*eps_view1_range)
            eps2 = _random.uniform(*eps_view2_range)
            x1 = gaussian_noise_uthar(x, eps1)
            x2 = gaussian_noise_uthar(x, eps2)
            feat1, feat2 = model(x1, x2, flag="unsupervised")
            losses = criterion(feat1, feat2)
            loss = losses["final-kde"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
            n += 1
        avg = total / max(1, n)
        history.append(avg)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[autofi-uthar-ssl] epoch {epoch+1}/{epochs} final-kde={avg:.4f}")
    return history


def linear_probe_autofi_uthar(
    model: AutoFiUTHARParallel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
) -> dict[str, float]:
    """AutoFi paper §IV-C K-shot calibration via linear probe."""
    model = model.to(device)
    optim = torch.optim.Adam(
        model.classifier.parameters(), lr=lr, weight_decay=weight_decay
    )
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            y1, y2 = model(x, x, flag="supervised")
            loss = ce(y1, y) + ce(y2, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    correct_1 = correct_2 = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            y1, y2 = model(x, x, flag="supervised")
            correct_1 += int((y1.argmax(dim=1) == y).sum().item())
            correct_2 += int((y2.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return {
        "accuracy_branch1": correct_1 / max(1, total),
        "accuracy_branch2": correct_2 / max(1, total),
        "accuracy": max(correct_1, correct_2) / max(1, total),
    }


def run_autofi_uthar(
    *,
    seed: int,
    ssl_epochs: int = 100,
    probe_epochs: int = 300,
    batch_size: int = 64,
    k_shot: int = 20,
    ut_har_root: str = "data/ut_har/UT_HAR",
    cache_dir: str = "data/widar3/cache",
) -> float:
    """Single-seed AutoFi run on UT-HAR. Returns top-1 accuracy on the test set.

    Paper §IV-C protocol: SSL pre-train on full UT-HAR train; K-shot
    calibration (K=20 -> 0.788 in paper Fig. 4) draws ``K`` labeled samples
    per class from train; eval on the canonical 500-sample test set.
    """
    import random
    import numpy as np
    from .ut_har import NUM_UT_HAR_CLASSES, UTHARDataset, ut_har_k_shot_indices

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = UTHARDataset(
        "train", root=ut_har_root, cache_path=f"{cache_dir}/ut_har-train.pt"
    )
    test_ds = UTHARDataset(
        "test", root=ut_har_root, cache_path=f"{cache_dir}/ut_har-test.pt"
    )
    train_labels = train_ds._y.numpy()
    k_idx = ut_har_k_shot_indices(
        train_labels, k=k_shot, num_classes=NUM_UT_HAR_CLASSES, seed=seed
    )
    probe_train_ds = UTHARDataset(
        "train",
        root=ut_har_root,
        indices=k_idx.tolist(),
        cache_path=f"{cache_dir}/ut_har-train.pt",
    )
    print(
        f"[autofi-uthar] ssl_n={len(train_ds)} probe_train_n={len(probe_train_ds)} "
        f"test_n={len(test_ds)} k_shot={k_shot}"
    )

    ssl_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    probe_train_loader = DataLoader(probe_train_ds, batch_size=batch_size, shuffle=True)
    probe_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AutoFiUTHARParallel(num_classes=NUM_UT_HAR_CLASSES, hidden_states=256)
    pretrain_autofi_uthar(model, ssl_loader, epochs=ssl_epochs, device=device)
    metrics = linear_probe_autofi_uthar(
        model, probe_train_loader, probe_test_loader, epochs=probe_epochs, device=device
    )
    print(
        f"[autofi-uthar] probe acc branch1={metrics['accuracy_branch1']:.4f} "
        f"branch2={metrics['accuracy_branch2']:.4f} max={metrics['accuracy']:.4f}"
    )
    return float(metrics["accuracy"])
