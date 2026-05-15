"""Augmentations for Slice 2 (static-component perturbation).

T2.5 ships `static_perturb` — the slice's signature augmentation:

    1. Decompose each batch sample into (static, dynamic).
    2. Permute the static components across the batch dimension.
    3. Recombine permuted-static + original-dynamic.

The intent: pre-training with this augmentation teaches the encoder that
the static (room) component is irrelevant to gesture identity, which
should improve cross-environment generalisation. Test on cross-environment
splits (Slice 2's natural target) once T2.6 / T2.7 run.

For completeness, this module also ships the hand-crafted-augmentation
baseline functions (`gaussian_noise`, `random_subcarrier_mask`,
`gaussian_then_mask`) so the slice's `simclr-handcrafted` comparison
mode is self-contained, matching Slice 1's and Slice 5's parameters
exactly so the team-paper rows sit on identical augmentation pipelines.
"""

from __future__ import annotations

import torch

from .decompose import DecompMethod, static_dynamic_split


def static_perturb(
    x: torch.Tensor,
    method: DecompMethod = "time-mean",
    cutoff_hz: float = 2.0,
    sample_rate_hz: float = 1000.0,
) -> torch.Tensor:
    """Static-component perturbation augmentation.

    Args:
        x: `(B, T, S, A)` batched CSI. Per-sample static-swap only makes
            sense in the batched path — the swap target needs another
            sample. A single sample `(T, S, A)` is returned unchanged.
        method, cutoff_hz, sample_rate_hz: forwarded to `static_dynamic_split`.

    Returns:
        Same shape as `x`. For `B ≥ 2`, each sample's static component is
        replaced by another sample's static (cyclic permutation by 1, so
        sample `i` receives sample `(i + 1) % B`'s static); dynamic is
        kept. The cyclic-permutation choice is deterministic given the
        batch — adequate for SimCLR because the two SimCLR views differ
        in their independent crops/augmentations of the *same* batch
        ordering, not because the perm itself is random per view.
    """
    if x.ndim == 3:
        return x
    if x.ndim != 4:
        raise ValueError(f"static_perturb expects (B, T, S, A); got ndim={x.ndim}")
    b = x.shape[0]
    if b < 2:
        return x

    static, dynamic = static_dynamic_split(
        x, method=method, cutoff_hz=cutoff_hz, sample_rate_hz=sample_rate_hz
    )
    # Cyclic permutation: sample i gets sample (i+1) % B's static.
    perm = torch.arange(b, device=x.device)
    perm = (perm + 1) % b
    return static[perm] + dynamic


# ---------------------------------------------------------------------------
# T2.6 hand-crafted-aug baseline (same as Slice 1 / Slice 5)
# Used as the comparison column for the team-paper figure.


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def random_subcarrier_mask(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
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
    return random_subcarrier_mask(gaussian_noise(x, sigma=sigma), p=p)
