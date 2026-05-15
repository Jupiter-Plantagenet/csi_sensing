"""End-to-end run for Slice 2.

Three modes via `--mode`:

- `mode=simclr-none` (T2.1 default): no augmentation, plumbing only.
- `mode=simclr-handcrafted` (T2.6): Gaussian + subcarrier mask — the
  comparison column.
- `mode=simclr-static-perturb` (T2.5): static-component perturbation —
  this slice's contribution.

With `--real`, swaps `StubCSI` for the cross-environment loader. The
cross-environment split is keyed by recording date: pass `--test-date`
to set the held-out date.

Run:
    python -m src.slices.chigozie.run                                 # stub, no aug
    python -m src.slices.chigozie.run --mode simclr-handcrafted       # stub baseline
    python -m src.slices.chigozie.run --mode simclr-static-perturb    # stub ours
    python -m src.slices.chigozie.run --mode simclr-static-perturb --real --test-date 20181128
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import gaussian_then_mask, static_perturb
from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI, Widar3CrossEnvironment
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_datasets(
    real: bool,
    seed: int,
    data_root: str,
    cache_dir: str,
    test_dates: list[str],
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if real:
        train_ds = Widar3CrossEnvironment(
            root=data_root,
            train=True,
            test_dates=test_dates,
            cache_path=f"{cache_dir}/chigozie_train.pt",
        )
        test_ds = Widar3CrossEnvironment(
            root=data_root,
            train=False,
            test_dates=test_dates,
            cache_path=f"{cache_dir}/chigozie_test.pt",
        )
        print(
            f"[chigozie] real Widar3.0 cross-environment; "
            f"train={len(train_ds)}, test={len(test_ds)}"
        )
    else:
        train_ds = StubCSI(num_samples=10, seed=seed)
        test_ds = StubCSI(num_samples=10, seed=seed + 1)
        print(f"[chigozie] stub data; train={len(train_ds)}, test={len(test_ds)}")
    return train_ds, test_ds


_MODE_AUGS = {
    "simclr-none": None,
    "simclr-handcrafted": gaussian_then_mask,
    "simclr-static-perturb": static_perturb,
}
_MODE_TAGS = {
    "simclr-none": "T2.1",
    "simclr-handcrafted": "T2.6 (baseline)",
    "simclr-static-perturb": "T2.5 (ours)",
}


def main(
    mode: str = "simclr-none",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    real: bool = False,
    data_root: str = "data/widar3/raw",
    cache_dir: str = "data/widar3/cache",
    test_dates: list[str] | None = None,
) -> float:
    if mode not in _MODE_AUGS:
        raise ValueError(f"unknown mode: {mode!r}; pick from {sorted(_MODE_AUGS)}")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, test_ds = _make_datasets(
        real, seed, data_root, cache_dir, test_dates or ["20181128"]
    )
    pretrain_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    encoder = TinyCNN(in_channels=CSI_S * CSI_A, feature_dim=128)
    tag = _MODE_TAGS[mode]
    print(f"[{tag}] encoder params: {count_parameters(encoder)}")

    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
    losses = pretrain_simclr(
        model,
        pretrain_loader,
        epochs=epochs,
        lr=1e-3,
        temperature=0.5,
        augment_fn=_MODE_AUGS[mode],
        device=device,
    )
    print(f"[{tag}] SSL pre-train losses: {[round(loss, 4) for loss in losses]}")

    acc = linear_probe(
        model.encoder, probe_train_loader, test_loader, device=device, seed=seed
    )
    chance = 1.0 / NUM_CLASSES
    print(f"[{tag}] linear-probe accuracy: {acc:.3f} (chance ~ {chance:.3f})")
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(_MODE_AUGS), default="simclr-none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--real", action="store_true")
    p.add_argument("--data-root", default="data/widar3/raw")
    p.add_argument("--cache-dir", default="data/widar3/cache")
    p.add_argument(
        "--test-date",
        action="append",
        dest="test_dates",
        default=None,
        help="Date string held out for the cross-environment test split. "
        "Repeat for multiple dates.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        real=args.real,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        test_dates=args.test_dates,
    )
