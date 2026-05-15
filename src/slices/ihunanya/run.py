"""End-to-end smoke run for Slice 4 (T4.1).

Wires StubCSI -> TinyCNN encoder -> SimCLR pre-training (no aug) ->
linear-probe eval. T4.1 only proves the pipeline runs; T4.3 will add the
coherence-bandwidth estimation, T4.5 the coherence-aware subcarrier masking
augmentation that this slice's actual contribution is.

Run:
    python -m src.slices.ihunanya.run
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrain_ds = StubCSI(num_samples=10, seed=seed)
    train_ds = StubCSI(num_samples=10, seed=seed + 1)
    test_ds = StubCSI(num_samples=10, seed=seed + 2)

    # drop_last=True so a singleton SimCLR batch doesn't collapse NT-Xent.
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    encoder = TinyCNN(in_channels=CSI_S * CSI_A, feature_dim=128)
    print(f"[T4.1] encoder params: {count_parameters(encoder)}")

    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    losses = pretrain_simclr(
        model,
        pretrain_loader,
        epochs=epochs,
        lr=1e-3,
        temperature=0.5,
        augment_fn=None,  # T4.5 will plug in the static-perturbation aug
        device=device,
    )
    print(f"[T4.1] pre-train losses: {[round(loss, 4) for loss in losses]}")

    acc = linear_probe(
        model.encoder, train_loader, test_loader, device=device, seed=seed
    )
    chance = 1.0 / NUM_CLASSES
    print(
        f"[T4.1] linear-probe accuracy on stub data: {acc:.3f} (chance ~ {chance:.3f})"
    )
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
