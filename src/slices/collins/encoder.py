"""Tiny 1D CNN encoder for CSI tensors.

Input is shaped (B, C, T) where C = subcarrier * antenna_pair (the channel axis)
and T is the time axis. The encoder outputs a fixed-length feature vector.

Channel widths 32 -> 64 -> 128 with kernel 3 give roughly 40K parameters for a
typical Widar3.0 input (C = 30 * 3 = 90), close to the project-wide ~50K target
in docs/07-experiment-scaffold.md.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
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


def reshape_csi_for_encoder(x: torch.Tensor) -> torch.Tensor:
    """Convert a CSI batch from (B, T, S, A) to (B, S*A, T).

    Datasets in this slice yield samples shaped (T, S, A); after the default
    PyTorch collate this becomes (B, T, S, A). The encoder expects channels
    first along the subcarrier-antenna axis, with time as the conv axis.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B, T, S, A); got shape {tuple(x.shape)}")
    b, t, s, a = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, s * a, t)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
