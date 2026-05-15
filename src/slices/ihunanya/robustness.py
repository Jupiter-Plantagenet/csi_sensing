"""Held-out subcarrier robustness sweep for Slice 4 (T4.6).

The slice's distinctive contribution to the team paper. After training
(supervised or SSL), evaluate the model on the test set while randomly
masking N of the S subcarriers at test time. Sweep N from 0 (no masking)
upward; plot accuracy vs N for the hand-crafted baseline and for the
coherence-aware-mask augmentation. A coherence-aware encoder should
degrade more gracefully — that's the claim.

This is the source of `papers/team/coherence-robustness-figure.png`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from .encoder import reshape_csi_for_encoder


@torch.no_grad()
def _extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    test_mask_n: int = 0,
    test_mask_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as `eval.extract_features` but applies a random subcarrier mask of
    width `test_mask_n` to every input batch first (zero-out, contiguous
    region per batch). Used for the robustness sweep.
    """
    encoder.eval()
    encoder.to(device)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    g = torch.Generator(device="cpu")
    if test_mask_seed is not None:
        g.manual_seed(test_mask_seed)
    for batch in loader:
        x, y = batch
        x = x.to(device).float()
        if test_mask_n > 0:
            s = x.shape[-2]
            max_start = max(0, s - test_mask_n)
            start = int(torch.randint(0, max_start + 1, (1,), generator=g).item())
            x = x.clone()
            x[..., start : start + test_mask_n, :] = 0
        x = reshape_csi_for_encoder(x)
        h = encoder(x)
        features.append(h.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def robustness_sweep(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    mask_widths: Iterable[int],
    device: str = "cpu",
    seed: int = 42,
) -> dict[int, float]:
    """Sweep test-time subcarrier-mask widths; return {width: accuracy}.

    Fits a linear classifier on the (un-masked) training features so the
    classifier is fair across mask widths; only the test-time features
    are perturbed. This isolates "is the encoder's representation robust
    to test-time subcarrier dropout?" from "did the classifier learn
    something brittle?".
    """
    train_x, train_y = _extract_features(
        encoder, train_loader, device=device, test_mask_n=0
    )
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_x, train_y)

    out: dict[int, float] = {}
    for n in mask_widths:
        test_x, test_y = _extract_features(
            encoder,
            test_loader,
            device=device,
            test_mask_n=int(n),
            test_mask_seed=seed + int(n),
        )
        pred = clf.predict(test_x)
        out[int(n)] = float(np.mean(pred == test_y))
    return out
