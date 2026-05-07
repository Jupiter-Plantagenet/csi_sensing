"""Linear-probe evaluation on a frozen encoder.

The evaluation protocol mandated by docs/07-experiment-scaffold.md: freeze the
SSL-pre-trained encoder, train a linear classifier on top of its features using
labelled data, report classifier accuracy on a held-out target-domain split.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from .encoder import reshape_csi_for_encoder


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    encoder.to(device)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        x, y = batch
        x = x.to(device).float()
        x = reshape_csi_for_encoder(x)
        h = encoder(x)
        features.append(h.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str = "cpu",
    max_iter: int = 1000,
    seed: int = 42,
) -> float:
    """Fit a linear classifier on frozen-encoder features; return test accuracy."""
    train_x, train_y = extract_features(encoder, train_loader, device=device)
    test_x, test_y = extract_features(encoder, test_loader, device=device)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return float(np.mean(pred == test_y))
