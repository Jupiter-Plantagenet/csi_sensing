"""End-to-end run for Slice 5.

T5.1 (default) wires StubCSI -> TinyCNN encoder -> linear classifier ->
cross-entropy training -> top-1 accuracy. Stub data, accuracy at chance.

T5.2 adds the `--real` flag to swap in `Widar3CrossSubject`. The same
training and eval code then produces a defensible cross-subject accuracy.

Run:
    python -m src.slices.josiah.run                     # stub data (T5.1)
    python -m src.slices.josiah.run --real --epochs 10  # real data (T5.2)
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI, Widar3CrossSubject
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
    real: bool = False,
    data_root: str = "data/widar3/raw",
    cache_dir: str = "data/widar3/cache",
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if real:
        train_ds = Widar3CrossSubject(
            root=data_root,
            train=True,
            cache_path=f"{cache_dir}/josiah_train.pt",
        )
        test_ds = Widar3CrossSubject(
            root=data_root,
            train=False,
            cache_path=f"{cache_dir}/josiah_test.pt",
        )
        print(
            f"[T5.2] real Widar3.0 cross-subject; "
            f"train={len(train_ds)}, test={len(test_ds)}"
        )
    else:
        train_ds = StubCSI(num_samples=10, seed=seed)
        test_ds = StubCSI(num_samples=10, seed=seed + 1)
        print(f"[T5.1] stub data; train={len(train_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SupervisedClassifier(
        in_channels=CSI_S * CSI_A, num_classes=NUM_CLASSES, feature_dim=128
    )
    print(f"[josiah] model params: {count_parameters(model)}")

    losses = train_supervised(
        model, train_loader, epochs=epochs, lr=1e-3, device=device
    )
    print(f"[josiah] train losses: {[round(loss, 4) for loss in losses]}")

    acc = evaluate(model, test_loader, device=device)
    chance = 1.0 / NUM_CLASSES
    tag = "T5.2" if real else "T5.1"
    print(f"[{tag}] supervised top-1 accuracy: {acc:.3f} (chance ~ {chance:.3f})")
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--real",
        action="store_true",
        help="Use real Widar3.0 data (T5.2); default is stub (T5.1).",
    )
    p.add_argument(
        "--data-root",
        default="data/widar3/raw",
        help="Directory containing Widar3.0 .dat files (recursive).",
    )
    p.add_argument(
        "--cache-dir",
        default="data/widar3/cache",
        help="Directory for cached parsed tensors.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        real=args.real,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
