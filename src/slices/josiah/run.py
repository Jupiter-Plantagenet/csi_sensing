"""End-to-end runs for Slice 5 project baselines.

This module intentionally contains only the project-owned baselines:

- `supervised` (T5.1 / T5.2): cross-entropy on the full classifier.
- `simclr-trivial` (T5.3): SimCLR with random crop only.
- `simclr-handcrafted` (T5.6): SimCLR with Gaussian noise + subcarrier mask.

Published AutoFi/CAPC reproductions are not exposed here until exact
implementations exist. Adapted approximations were removed so they cannot be
mistaken for published-baseline reproductions.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import gaussian_then_mask, random_crop
from .data import NUM_CLASSES, StubCSI, Widar3CrossSubject
from .encoder import SupervisedClassifier, TinyCNN, count_parameters
from .eval import evaluate, linear_probe, train_supervised
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _infer_sample_shape(ds: torch.utils.data.Dataset) -> tuple[int, int, int]:
    sample, _ = ds[0]
    if sample.ndim != 3:
        raise ValueError(f"expected sample (T, S, A); got {tuple(sample.shape)}")
    return int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2])


def _infer_in_channels(ds: torch.utils.data.Dataset) -> int:
    _t, s, a = _infer_sample_shape(ds)
    return s * a


def _make_datasets(
    real: bool,
    seed: int,
    data_root: str,
    cache_dir: str,
    representation: str,
    time_steps: int,
    max_files: int | None,
    receivers: list[int] | None = None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if real:
        rx_tag = "r" + "".join(str(r) for r in receivers) if receivers else "rcanon"
        cache_tag = f"josiah-{representation}-T{time_steps}-{rx_tag}-max{max_files or 'all'}"
        train_ds = Widar3CrossSubject(
            root=data_root,
            train=True,
            cache_path=f"{cache_dir}/{cache_tag}-train.pt",
            representation=representation,
            time_steps=time_steps,
            max_files=max_files,
            receivers=receivers,
        )
        test_ds = Widar3CrossSubject(
            root=data_root,
            train=False,
            cache_path=f"{cache_dir}/{cache_tag}-test.pt",
            representation=representation,
            time_steps=time_steps,
            max_files=max_files,
            receivers=receivers,
        )
        print(
            f"[josiah] real Widar3.0 cross-subject "
            f"(receivers={receivers or 'canonical[1]'}); "
            f"train={len(train_ds)}, test={len(test_ds)}"
        )
    else:
        train_ds = StubCSI(num_samples=10, seed=seed)
        test_ds = StubCSI(num_samples=10, seed=seed + 1)
        print(f"[josiah] stub data; train={len(train_ds)}, test={len(test_ds)}")
    return train_ds, test_ds


def main(
    mode: str = "supervised",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    real: bool = False,
    data_root: str = "data/widar3/raw",
    cache_dir: str = "data/widar3/cache",
    representation: str = "real-imag",
    time_steps: int = 200,
    max_files: int | None = None,
    receivers: list[int] | None = None,
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, test_ds = _make_datasets(
        real, seed, data_root, cache_dir, representation, time_steps, max_files,
        receivers=receivers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=(mode != "supervised")
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if mode == "supervised":
        in_channels = _infer_in_channels(train_ds)
        model = SupervisedClassifier(
            in_channels=in_channels, num_classes=NUM_CLASSES, feature_dim=128
        )
        print(f"[josiah] supervised model params: {count_parameters(model)}")
        losses = train_supervised(
            model, train_loader, epochs=epochs, lr=1e-3, device=device
        )
        print(f"[josiah] train losses: {[round(loss, 4) for loss in losses]}")
        acc = evaluate(model, test_loader, device=device)
        print(f"[josiah] supervised top-1 accuracy: {acc:.3f}")
        return acc

    if mode in ("simclr-trivial", "simclr-handcrafted"):
        tag = "T5.3" if mode == "simclr-trivial" else "T5.6"
        augment_fn = random_crop if mode == "simclr-trivial" else gaussian_then_mask
        in_channels = _infer_in_channels(train_ds)
        encoder = TinyCNN(in_channels=in_channels, feature_dim=128)
        print(f"[{tag}] encoder params: {count_parameters(encoder)}")
        ssl_model = SimCLR(encoder, feature_dim=128, projection_dim=64)
        losses = pretrain_simclr(
            ssl_model,
            train_loader,
            epochs=epochs,
            lr=1e-3,
            temperature=0.5,
            augment_fn=augment_fn,
            device=device,
        )
        print(f"[{tag}] SSL pre-train losses: {[round(loss, 4) for loss in losses]}")

        probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        acc = linear_probe(
            ssl_model.encoder,
            probe_train_loader,
            test_loader,
            device=device,
            seed=seed,
        )
        print(f"[{tag}] linear-probe accuracy: {acc:.3f}")
        return acc

    raise ValueError(f"unknown mode: {mode!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["supervised", "simclr-trivial", "simclr-handcrafted"],
        default="supervised",
        help=(
            "Project baseline mode: supervised (T5.1/T5.2), "
            "simclr-trivial (T5.3), or simclr-handcrafted (T5.6). "
            "AutoFi/CAPC exact reproductions require separate exact code."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--real",
        action="store_true",
        help="Use real Widar3.0 data; default is stub.",
    )
    p.add_argument("--data-root", default="data/widar3/raw")
    p.add_argument("--cache-dir", default="data/widar3/cache")
    p.add_argument("--representation", choices=["real-imag", "magnitude"], default="real-imag")
    p.add_argument("--time-steps", type=int, default=200)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--receivers",
        default=None,
        help="CSV receiver IDs e.g. 1,2,3. Default: canonical [1] per roadmap.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    receivers = (
        [int(r) for r in args.receivers.split(",") if r]
        if args.receivers
        else None
    )
    main(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        real=args.real,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        representation=args.representation,
        time_steps=args.time_steps,
        max_files=args.max_files,
        receivers=receivers,
    )
