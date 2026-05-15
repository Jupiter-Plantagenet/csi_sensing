"""SimCLR wrapper, projection head, and NT-Xent loss for Slice 2.

Ported from Slice 1 per the slice-independence convention. Same SSL
machinery; the slice's distinctive contribution is the augmentation
(static-component perturbation) plugged in via `augment_fn`.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import reshape_csi_for_encoder

AugmentFn = Callable[[torch.Tensor], torch.Tensor]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(
        self, encoder: nn.Module, feature_dim: int, projection_dim: int = 64
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(
            feature_dim, hidden_dim=feature_dim, out_dim=projection_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.projection(h)


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))

    targets = torch.arange(2 * b, device=z.device)
    targets = (targets + b) % (2 * b)

    return F.cross_entropy(sim, targets)


def make_views(
    batch: torch.Tensor,
    augment_fn: Optional[AugmentFn] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the SimCLR (view1, view2) pair."""
    if augment_fn is None:
        v1, v2 = batch, batch
    else:
        v1, v2 = augment_fn(batch), augment_fn(batch)
    return reshape_csi_for_encoder(v1), reshape_csi_for_encoder(v2)


def pretrain_simclr(
    model: SimCLR,
    loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    temperature: float = 0.5,
    augment_fn: Optional[AugmentFn] = None,
    device: str = "cpu",
) -> list[float]:
    """Run SimCLR pre-training. Returns per-epoch mean losses."""
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []
    for _ in range(epochs):
        batch_losses: list[float] = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            v1, v2 = make_views(x, augment_fn=augment_fn)
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))
    return epoch_losses
