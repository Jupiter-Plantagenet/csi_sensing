"""End-to-end run for Slice 3.

Wires the dataset -> TinyCNN encoder -> SimCLR pre-training -> linear-probe
eval. Selects between StubCSI (T3.1, runs without any data) and the real
Widar3CrossSubject loader (T3.2 onward, requires extracted Widar3.0 under
``--data-root``).

Run:
    python -m src.slices.collins.run                      # stub data
    python -m src.slices.collins.run \\
        --data-root data/widar3/extracted \\
        --cache-dir data/widar3/cache \\
        --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import AugmentFn, make_generic_aug, make_phase_noise_aug
from .data import NUM_CLASSES, StubCSI, Widar3CrossSubject
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _infer_in_channels(ds: Dataset) -> int:
    """Read one sample and infer encoder in_channels = S * A."""
    sample, _ = ds[0]
    if sample.ndim != 3:
        raise ValueError(
            f"expected each sample to be (T, S, A); got shape {tuple(sample.shape)}"
        )
    _, s, a = sample.shape
    return s * a


def main(
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    data_root: str | None = None,
    cache_dir: str | None = None,
    max_files: int | None = None,
    phase_profile: str | None = None,
) -> dict:
    """Run one full pipeline (pre-train + linear probe). Returns a dict with
    accuracy, loss curve, and the resolved config — `compare.py` (T3.6) reads
    this directly to log a paired comparison.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augment_fn: Optional[AugmentFn] = None
    augment_name: str = "none"

    if data_root is None:
        tag = "T3.1-stub"
        pretrain_ds: Dataset = StubCSI(num_samples=16, seed=seed)
        train_ds: Dataset = StubCSI(num_samples=16, seed=seed + 1)
        test_ds: Dataset = StubCSI(num_samples=16, seed=seed + 2)
    else:
        pretrain_ds = Widar3CrossSubject(
            data_root,
            train=True,
            cache_dir=cache_dir,
            max_files=max_files,
        )
        train_ds = Widar3CrossSubject(
            data_root,
            train=True,
            cache_dir=cache_dir,
            max_files=max_files,
        )
        test_ds = Widar3CrossSubject(
            data_root,
            train=False,
            cache_dir=cache_dir,
            max_files=max_files,
        )
        # Augmentation choice: phase-noise (T3.5) when a profile is provided,
        # otherwise the generic baseline (T3.6's baseline). Both factories live
        # in augmentations.py; we just hand SimCLR the resulting callable.
        if phase_profile is not None:
            tag = "T3.5-widar3-phase-noise"
            augment_fn = make_phase_noise_aug(phase_profile)
            augment_name = "phase_noise"
        else:
            tag = "T3.2-widar3"
            augment_fn = make_generic_aug(sigma=0.3, subcarrier_drop_prob=0.15)
            augment_name = "generic_baseline"
        print(
            f"[{tag}] dataset sizes: pretrain={len(pretrain_ds)}, "
            f"train={len(train_ds)}, test={len(test_ds)}"
        )

    # drop_last=True avoids singleton SimCLR batches: a B=1 batch makes
    # NT-Xent collapse to zero loss after diagonal masking, contributing
    # no contrastive gradient and skewing the reported epoch mean.
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_channels = _infer_in_channels(pretrain_ds)
    encoder = TinyCNN(in_channels=in_channels, feature_dim=128)
    encoder_params = count_parameters(encoder)
    print(f"[{tag}] encoder in_channels={in_channels}, params={encoder_params}")

    model = SimCLR(encoder, feature_dim=128, projection_dim=64)
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
        f"[{tag}] linear-probe accuracy: {acc:.3f} "
        f"(chance ~ {1.0 / NUM_CLASSES:.3f})"
    )

    return {
        "tag": tag,
        "augment": augment_name,
        "accuracy": acc,
        "pretrain_losses": [float(round(loss, 6)) for loss in losses],
        "encoder_in_channels": int(in_channels),
        "encoder_params": int(encoder_params),
        "dataset_sizes": {
            "pretrain": len(pretrain_ds),
            "train": len(train_ds),
            "test": len(test_ds),
        },
        "config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_root": data_root,
            "cache_dir": cache_dir,
            "max_files": max_files,
            "phase_profile": phase_profile,
            "device": device,
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="path to extracted Widar3.0 (parent of YYYYMMDD/userN/*.dat). "
        "If omitted, uses StubCSI.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="cache parsed/normalized .dat tensors to .npy here.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="cap the number of .dat files used in train and test (debug).",
    )
    p.add_argument(
        "--phase-profile",
        type=str,
        default=None,
        help="path to a T3.3 phase_profile.npz; when set, replaces the "
        "generic baseline aug with calibrated phase-noise injection (T3.5).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        phase_profile=args.phase_profile,
    )
