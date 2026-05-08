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
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import NUM_CLASSES, StubCSI, Widar3CrossSubject
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .ssl import SimCLR, pretrain_simclr


def make_generic_aug(
    sigma: float = 0.3,
    subcarrier_drop_prob: float = 0.15,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gaussian noise + random subcarrier dropout for SimCLR view generation.

    SimCLR with `augment_fn=None` produces identical views, which collapses
    the contrastive task to a trivial solution and leaves the encoder with
    nothing to learn. This stopgap matches the "generic baseline" augmentation
    in docs/03 (Gaussian noise + random subcarrier mask) so T3.6 can simply
    reuse it as the baseline once the phase-noise augmentation lands in T3.5.

    Input expected shape: (B, T, S, A) real-valued (z-scored CSI).
    """

    def _aug(x: torch.Tensor) -> torch.Tensor:
        x = x + torch.randn_like(x) * sigma
        if subcarrier_drop_prob > 0.0:
            b, _t, s, _a = x.shape
            keep = (torch.rand(b, 1, s, 1, device=x.device) > subcarrier_drop_prob).to(
                x.dtype
            )
            x = x * keep
        return x

    return _aug


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
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    if data_root is None:
        tag = "T3.1-stub"
        pretrain_ds: Dataset = StubCSI(num_samples=16, seed=seed)
        train_ds: Dataset = StubCSI(num_samples=16, seed=seed + 1)
        test_ds: Dataset = StubCSI(num_samples=16, seed=seed + 2)
    else:
        tag = "T3.2-widar3"
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
        # Stopgap augmentation so SimCLR has a non-trivial task; replaced by
        # the calibrated phase-noise augmentation in T3.5.
        augment_fn = make_generic_aug(sigma=0.3, subcarrier_drop_prob=0.15)
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
    print(
        f"[{tag}] encoder in_channels={in_channels}, "
        f"params={count_parameters(encoder)}"
    )

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
    return acc


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
    )
