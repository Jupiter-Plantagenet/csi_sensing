"""End-to-end run for Slice 4.

Three modes via `--mode`:

- `simclr-none` (T4.1): no augmentation, plumbing only.
- `simclr-handcrafted` (the comparison baseline): Gaussian + random
  subcarrier mask.
- `simclr-coherent-mask` (T4.5, ours): contiguous block of width
  ≈ coherence bandwidth, estimated from the pre-training data.

`--sweep` adds the held-out-subcarrier robustness sweep (T4.6) after
pre-training and linear probe.

Run:
    python -m src.slices.ihunanya.run                              # plumbing
    python -m src.slices.ihunanya.run --mode simclr-handcrafted    # baseline
    python -m src.slices.ihunanya.run --mode simclr-coherent-mask  # ours
    python -m src.slices.ihunanya.run --mode simclr-coherent-mask --sweep
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import coherent_block_mask, gaussian_then_mask
from .coherence import estimate_coherence_bandwidth_subcarriers
from .data import CSI_A, CSI_S, NUM_CLASSES, StubCSI
from .encoder import TinyCNN, count_parameters
from .eval import linear_probe
from .robustness import robustness_sweep
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_aug(mode: str, block_width: int):
    if mode == "simclr-none":
        return None
    if mode == "simclr-handcrafted":
        return gaussian_then_mask
    if mode == "simclr-coherent-mask":

        def _aug(x: torch.Tensor) -> torch.Tensor:
            return coherent_block_mask(x, block_width=block_width)

        _aug.__name__ = f"coherent_block_mask_w{block_width}"
        return _aug
    raise ValueError(f"unknown mode: {mode!r}")


_MODE_TAGS = {
    "simclr-none": "T4.1",
    "simclr-handcrafted": "T4.6 (baseline)",
    "simclr-coherent-mask": "T4.5 (ours)",
}


def main(
    mode: str = "simclr-none",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    sweep: bool = False,
    sweep_widths: list[int] | None = None,
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrain_ds = StubCSI(num_samples=10, seed=seed)
    train_ds = StubCSI(num_samples=10, seed=seed + 1)
    test_ds = StubCSI(num_samples=10, seed=seed + 2)

    sample_for_estimate = torch.stack([pretrain_ds[i][0] for i in range(len(pretrain_ds))])
    coh_bw = estimate_coherence_bandwidth_subcarriers(sample_for_estimate)
    print(f"[ihunanya] estimated coherence bandwidth: {coh_bw} subcarriers")

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

    if sweep:
        widths = sweep_widths or [0, 2, 5, 10, 15]
        print(f"[T4.6 robustness sweep] mask widths: {widths}")
        sweep_results = robustness_sweep(
            model.encoder,
            train_loader,
            test_loader,
            mask_widths=widths,
            device=device,
            seed=seed,
        )
        for n, sweep_acc in sweep_results.items():
            print(f"[T4.6 robustness sweep] mask={n:>3d}: acc={sweep_acc:.3f}")
    return acc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(_MODE_TAGS), default="simclr-none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sweep", action="store_true", help="Run T4.6 robustness sweep")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sweep=args.sweep,
    )
