"""Supervised train + accuracy eval for Slice 5.

The Slice 5 supervised baseline trains a `SupervisedClassifier` end-to-end
with cross-entropy against the gesture label. No SSL pre-training, no
augmentation other than what's needed to batch the data. This is the floor
of the comparison-to-conventional-solutions: every SSL method we report
later has to beat the supervised baseline to be worth anything.

`train_supervised` returns the per-epoch training-loss trajectory; `evaluate`
returns the test-set classification accuracy.
"""

from __future__ import annotations

import torch
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
