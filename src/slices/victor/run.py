"""End-to-end run for Slice 6 (composability of Doppler + coherence-mask).

Modes via `--mode`:

- `simclr-none` (T6.1): no augmentation, plumbing only.
- `simclr-handcrafted` (baseline): Gaussian + random subcarrier mask.
- `simclr-doppler` (T6.3): Doppler time warp alone.
- `simclr-coherent-mask` (T6.3): coherent block mask alone, width
  estimated from the pre-training data.
- `simclr-combined` (T6.5): Doppler then coherent mask, sequential
  composition on the same view.

The composability study (T6.5–T6.6) runs the four non-`none` modes and
computes the interaction term `combined - (doppler + coherent_mask)`.

Run:
    python -m src.slices.victor.run                              # plumbing
    python -m src.slices.victor.run --mode simclr-handcrafted
    python -m src.slices.victor.run --mode simclr-doppler
    python -m src.slices.victor.run --mode simclr-coherent-mask
    python -m src.slices.victor.run --mode simclr-combined
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import (
    coherent_block_mask,
    doppler_then_coherent_mask,
    doppler_warp,
    gaussian_then_mask,
)
from .coherence import estimate_coherence_bandwidth_subcarriers
from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_MODE_TAGS = {
    "simclr-none": "T6.1",
    "simclr-handcrafted": "T6.6 (baseline)",
    "simclr-doppler": "T6.4 (doppler)",
    "simclr-coherent-mask": "T6.4 (coh-mask)",
    "simclr-combined": "T6.5 (combined)",
}


def _build_aug(mode: str, block_width: int):
    if mode == "simclr-none":
        return None
    if mode == "simclr-handcrafted":
        return gaussian_then_mask
    if mode == "simclr-doppler":
        return doppler_warp
    if mode == "simclr-coherent-mask":

        def _aug(x: torch.Tensor) -> torch.Tensor:
            return coherent_block_mask(x, block_width=block_width)

        _aug.__name__ = f"coherent_block_mask_w{block_width}"
        return _aug
    if mode == "simclr-combined":

        def _aug(x: torch.Tensor) -> torch.Tensor:
            return doppler_then_coherent_mask(x, block_width=block_width)

        _aug.__name__ = f"doppler_then_coherent_mask_w{block_width}"
        return _aug
    raise ValueError(f"unknown mode: {mode!r}")


def main(
    mode: str = "simclr-none",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrain_ds = StubCSI(num_samples=10, seed=seed)
    train_ds = StubCSI(num_samples=10, seed=seed + 1)
    test_ds = StubCSI(num_samples=10, seed=seed + 2)

    sample_for_estimate = torch.stack([pretrain_ds[i][0] for i in range(len(pretrain_ds))])
    coh_bw = estimate_coherence_bandwidth_subcarriers(sample_for_estimate)
    print(f"[victor] estimated coherence bandwidth: {coh_bw} subcarriers")

    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    encoder = TinyCNN(in_channels=CSI_S * CSI_A, feature_dim=128)
    tag = _MODE_TAGS[mode]
    print(f"[{tag}] encoder params: {count_parameters(encoder)}")

    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    augment_fn = _build_aug(mode, block_width=coh_bw)
    losses = pretrain_simclr(
        model,
        pretrain_loader,
        epochs=epochs,
        lr=1e-3,
        temperature=0.5,
        augment_fn=augment_fn,
        device=device,
    )
    print(f"[{tag}] pre-train losses: {[round(loss, 4) for loss in losses]}")

    acc = linear_probe(
        model.encoder, train_loader, test_loader, device=device, seed=seed
    )
    print(
        f"[{tag}] linear-probe accuracy: {acc:.3f} (chance ~ {1 / NUM_CLASSES:.3f})"
    )
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(_MODE_TAGS), default="simclr-none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
