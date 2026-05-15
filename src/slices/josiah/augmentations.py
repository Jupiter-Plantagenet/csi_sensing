"""Augmentation functions for Slice 5's SSL baselines.

- `random_crop` (T5.3) — trivial-augmentation baseline. Tests "does SSL
  help at all on CSI, even with the dumbest augmentation?"
- `gaussian_noise` + `random_subcarrier_mask` + `gaussian_then_mask` (T5.6)
  — hand-crafted-augmentation baseline. This is the comparison-column row
  every physics-informed augmentation slice (1, 2, 4, 6) measures against.
  Ported from Slice 1 per the slice-independence rule so the comparison
  is apples-to-apples (same augmentation parameters).

Each augmentation accepts a batch shaped `(B, T, S, A)` or a single sample
`(T, S, A)` and returns the same shape. Two independent calls produce two
independent views; see `ssl.make_views`.
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


# ---------------------------------------------------------------------------
# T5.6 hand-crafted-augmentation baseline
# Gaussian noise + random subcarrier masking, ported from Slice 1. Same
# parameters as Slice 1 so the comparison row sits on identical augmentations.


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Additive Gaussian noise on every CSI element.

    `sigma` matches Slice 1's default. Pick it small relative to the typical
    CSI magnitude after the projection / normalisation upstream.
    """
    return x + torch.randn_like(x) * sigma


def random_subcarrier_mask(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """Zero out a random fraction `p` of subcarriers per sample.

    Operates on `(T, S, A)` or `(B, T, S, A)`. In the batched path each
    sample gets its own independent mask so two calls produce two views
    with different masked-out subcarriers — the SimCLR view-pair recipe.
    """
    if x.ndim == 3:
        s = x.shape[1]
        keep = torch.rand(s, device=x.device) >= p
        return x * keep.view(1, s, 1).to(x.dtype)
    if x.ndim == 4:
        b, _, s, _ = x.shape
        keep = torch.rand(b, s, device=x.device) >= p
        return x * keep.view(b, 1, s, 1).to(x.dtype)
    raise ValueError(
        f"random_subcarrier_mask expects (T, S, A) or (B, T, S, A); "
        f"got {tuple(x.shape)}"
    )


def gaussian_then_mask(
    x: torch.Tensor, sigma: float = 0.05, p: float = 0.15
) -> torch.Tensor:
    """T5.6 default view augmentation: Gaussian noise then subcarrier mask.

    Used symmetrically (both views call this) to produce the hand-crafted-
    augmentation baseline. The composition order — noise first, mask
    second — matches Slice 1's `gaussian_then_mask` so the comparison
    rows are byte-identical augmentation pipelines, only the encoder
    pre-training augmentation set differs across rows.
    """
    return random_subcarrier_mask(gaussian_noise(x, sigma=sigma), p=p)
