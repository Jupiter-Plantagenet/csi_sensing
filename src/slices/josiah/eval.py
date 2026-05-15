"""Evaluation routines for Slice 5.

Two protocols, both in the project conventions:

- **Supervised** (`train_supervised` + `evaluate`): cross-entropy on the
  full classifier, reported via top-1 test accuracy. The supervised
  baseline (T5.2) uses this and only this.

- **Linear probe** (`linear_probe`): freeze the SSL-pre-trained encoder,
  fit a logistic-regression classifier on its features, return test-set
  accuracy. Used by every SSL row (T5.3 trivial-aug, T5.6 hand-crafted-aug,
  T5.4 AutoFi, T5.5 CAPC).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from .encoder import SupervisedClassifier, reshape_csi_for_encoder


def train_supervised(
    model: SupervisedClassifier,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Cross-entropy training over `epochs` epochs. Returns mean loss per epoch."""
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for x, y in train_loader:
            x = reshape_csi_for_encoder(x).to(device)
            y = y.to(device).long()
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        history.append(sum(epoch_losses) / max(1, len(epoch_losses)))
    return history


@torch.no_grad()
def evaluate(
    model: SupervisedClassifier,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Top-1 classification accuracy on `test_loader`. Returns a float in [0, 1]."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = reshape_csi_for_encoder(x).to(device)
        y = y.to(device).long()
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    if total == 0:
        return 0.0
    return correct / total


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the (frozen) encoder over `loader`, returning feature/label arrays."""
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
    """Fit a linear classifier on frozen-encoder features; return test accuracy.

    Matches Slice 1's protocol: scikit-learn's LogisticRegression on the
    encoder's `(B, feature_dim)` outputs, evaluated as top-1 accuracy.
    """
    train_x, train_y = extract_features(encoder, train_loader, device=device)
    test_x, test_y = extract_features(encoder, test_loader, device=device)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return float(np.mean(pred == test_y))
