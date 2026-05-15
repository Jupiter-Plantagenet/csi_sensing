"""End-to-end smoke run for Slice 1.

Wires StubCSI -> TinyCNN encoder -> SimCLR pre-training -> linear-probe eval.
T1.1 only proves the pipeline runs; the printed accuracy is not meaningful.

Run:
    python -m src.slices.george.run
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import (
    doppler_warp,
    gaussian_noise,
    gaussian_then_mask,
    random_subcarrier_mask,
)
from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI, Widar3CrossSubject
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr

AUGMENTATIONS = {
    "none": None,
    "gaussian": gaussian_noise,
    "mask": random_subcarrier_mask,
    "generic": gaussian_then_mask,
    "doppler": doppler_warp,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    aug: str = "none",
    real: bool = False,
    data_root: str = "data/widar3/raw",
    cache_dir: str = "data/widar3/cache",
) -> float:
    if aug not in AUGMENTATIONS:
        raise ValueError(
            f"unknown augmentation {aug!r}; pick from {sorted(AUGMENTATIONS)}"
        )
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if real:
        pretrain_ds = Widar3CrossSubject(
            root=data_root, train=True, cache_path=f"{cache_dir}/george_train.pt"
        )
        train_ds = pretrain_ds
        test_ds = Widar3CrossSubject(
            root=data_root, train=False, cache_path=f"{cache_dir}/george_test.pt"
        )
        print(f"[T1.1] real Widar3.0 cross-subject; train={len(train_ds)}, test={len(test_ds)}")
    else:
        pretrain_ds = StubCSI(num_samples=10, seed=seed)
        train_ds = StubCSI(num_samples=10, seed=seed + 1)
        test_ds = StubCSI(num_samples=10, seed=seed + 2)

    # drop_last=True avoids singleton SimCLR batches: a B=1 batch makes
    # NT-Xent collapse to zero loss after diagonal masking, contributing
    # no contrastive gradient and skewing the reported epoch mean.
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    encoder = TinyCNN(in_channels=CSI_S * CSI_A, feature_dim=128)
    print(f"[T1.1] encoder params: {count_parameters(encoder)}")

    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    augment_fn = AUGMENTATIONS[aug]
    losses = pretrain_simclr(
        model,
        pretrain_loader,
        epochs=epochs,
        lr=1e-3,
        temperature=0.5,
        augment_fn=augment_fn,
        device=device,
    )
    print(f"[T1.1] aug={aug!r} pre-train losses: {[round(loss, 4) for loss in losses]}")

    acc = linear_probe(
        model.encoder, train_loader, test_loader, device=device, seed=seed
    )
    print(
        f"[T1.1] linear-probe accuracy on stub data: {acc:.3f} (chance ~ {1.0 / NUM_CLASSES:.3f})"
    )
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--aug", choices=sorted(AUGMENTATIONS), default="none")
    p.add_argument("--real", action="store_true", help="Use real Widar3.0 data.")
    p.add_argument("--data-root", default="data/widar3/raw")
    p.add_argument("--cache-dir", default="data/widar3/cache")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        aug=args.aug,
        real=args.real,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
