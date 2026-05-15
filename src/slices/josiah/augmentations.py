"""Augmentation functions for Slice 5's SSL baselines.

T5.3 uses `random_crop` only — the trivial-augmentation baseline that
tests "does SSL help at all on CSI?" T5.6 will add `gaussian_noise` and
`random_subcarrier_mask` for the hand-crafted-augmentation baseline (the
comparison column for slices 1, 2, 4, 6).

Each augmentation accepts a batch shaped `(B, T, S, A)` or a single sample
`(T, S, A)` and returns the same shape (cropping is followed by zero-pad
back to the original `T` so the encoder's batch dim stays consistent).
Two independent calls produce two independent views; see `ssl.make_views`.
"""

from __future__ import annotations

import torch


def random_crop(x: torch.Tensor, crop_ratio: float = 0.7) -> torch.Tensor:
    """Random temporal crop, padded back to the original length.

    Sample a start index per call (shared across the batch for simplicity)
    so the two SimCLR views differ by their random crops. The cropped
    segment is zero-padded back to the original `T` to keep the encoder's
    input length stable across batches.

    Args:
        x: `(B, T, S, A)` or `(T, S, A)`.
        crop_ratio: fraction of `T` retained per view. 0.7 → 30% of the
            time axis is zero-padded, leaving the encoder a 70%-length
            random window.

    Returns:
        Same shape as `x`; cropped window placed at the start, rest zeroed.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"expected (B, T, S, A) or (T, S, A); got ndim={x.ndim}")
    t = x.shape[-3]
    t_crop = max(1, int(t * crop_ratio))
    max_start = t - t_crop
    start = int(torch.randint(0, max_start + 1, (1,)).item())
    if x.ndim == 4:
        cropped = x[:, start : start + t_crop, :, :]
        out = torch.zeros_like(x)
        out[:, :t_crop, :, :] = cropped
    else:
        cropped = x[start : start + t_crop, :, :]
        out = torch.zeros_like(x)
        out[:t_crop, :, :] = cropped
    return out
