"""Tiny 1D CNN encoder + supervised classifier head for Slice 5.

Same encoder architecture as Slice 1 (see `src/slices/george/encoder.py`) —
matching the project-wide convention in `docs/07-experiment-scaffold.md`.
The slice owns its copy per the slice-independence rule in `docs/08`
section 2, so this module does not import from Slice 1.

For Slice 5 the encoder is paired with a linear classifier head and trained
end-to-end with cross-entropy on labelled cross-subject data. The encoder's
parameter count and feature dim are kept identical to Slice 1's so the
encoder-vs-augmentation comparison in the team's results figure is on like
terms (same backbone capacity, only the training objective differs).
"""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """3-layer 1D CNN encoder. Input `(B, C, T)`, output `(B, feature_dim)`."""

    def __init__(self, in_channels: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedClassifier(nn.Module):
    """Encoder + linear classifier head, trained jointly with cross-entropy."""

    def __init__(
        self, in_channels: int, num_classes: int, feature_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = TinyCNN(in_channels=in_channels, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def reshape_csi_for_encoder(x: torch.Tensor) -> torch.Tensor:
    """Convert a CSI batch from `(B, T, S, A)` to `(B, S*A, T)`."""
    if x.ndim != 4:
        raise ValueError(f"expected (B, T, S, A); got shape {tuple(x.shape)}")
    b, t, s, a = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, s * a, t)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
