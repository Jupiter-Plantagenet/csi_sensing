"""End-to-end smoke run for Slice 5 (T5.1).

Wires StubCSI -> TinyCNN encoder -> linear classifier -> cross-entropy
training -> top-1 accuracy. T5.1 only proves the pipeline runs; the
printed accuracy is not meaningful (chance level on random labels).

T5.2 will replace the stub data with a real Widar3.0 cross-subject loader.
T5.6 will be the hand-crafted-aug SimCLR baseline that the project's
physics-informed augmentations have to beat.

Run:
    python -m src.slices.josiah.run
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI
from .encoder import SupervisedClassifier, count_parameters
from .eval import evaluate, train_supervised


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

    train_ds = StubCSI(num_samples=10, seed=seed)
    test_ds = StubCSI(num_samples=10, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SupervisedClassifier(
        in_channels=CSI_S * CSI_A, num_classes=NUM_CLASSES, feature_dim=128
    )
    print(f"[T5.1] model params: {count_parameters(model)}")

    losses = train_supervised(
        model, train_loader, epochs=epochs, lr=1e-3, device=device
    )
    print(f"[T5.1] supervised train losses: " f"{[round(loss, 4) for loss in losses]}")

    acc = evaluate(model, test_loader, device=device)
    chance = 1.0 / NUM_CLASSES
    print(
        f"[T5.1] supervised top-1 accuracy on stub data: {acc:.3f} "
        f"(chance ~ {chance:.3f}; not meaningful - stub labels are random)"
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
