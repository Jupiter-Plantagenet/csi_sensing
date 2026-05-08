"""Augmentations for Slice 3 SimCLR pre-training.

Two factories live here, both returning a ``Callable[[Tensor], Tensor]`` that
accepts a real-valued CSI batch shaped ``(B, T, S, 2*A)`` — the real-imag
stacking convention from ``data.Widar3CrossSubject``: channels ``[0..A-1]``
hold the real parts of the antennas and channels ``[A..2A-1]`` hold the
imaginary parts.

- ``make_generic_aug`` — Gaussian noise + random subcarrier dropout. The
  project-wide "generic baseline" augmentation (docs/03 §5.3). T3.6 uses
  this as the baseline that phase-noise injection has to beat.
- ``make_phase_noise_aug`` — calibrated phase-noise injection per the T3.3
  per-subcarrier Gaussian profile. The slice's headline augmentation.

Both factories take their hyperparameters at construction time so the SimCLR
loop in ``ssl.py`` only ever sees a stateless ``augment_fn(batch) -> batch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from .phase_profile import load_profile

AugmentFn = Callable[[torch.Tensor], torch.Tensor]


def make_generic_aug(
    sigma: float = 0.3,
    subcarrier_drop_prob: float = 0.15,
) -> AugmentFn:
    """Gaussian noise + random subcarrier dropout. T3.6 baseline.

    Mirrors the "generic baseline" augmentation in docs/03: additive Gaussian
    noise with std ``sigma`` plus i.i.d. per-subcarrier Bernoulli dropout with
    probability ``subcarrier_drop_prob``. Same noise/mask realization is
    broadcast across time and antenna axes so a dropped subcarrier is dropped
    for the whole window — the way frequency-selective fading actually
    behaves on a coherence time-scale much longer than the packet rate.
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


def make_phase_noise_aug(
    profile_path: str | Path,
) -> AugmentFn:
    """Phase-noise injection from a fitted T3.3 profile.

    Loads ``(mu, sigma)`` from ``profile_path`` (the ``.npz`` produced by
    ``python -m src.slices.collins.phase_profile``). Each augmentation call
    samples per-``(B, T, S, A)`` element phase offsets ``phi ~ N(mu_k,
    sigma_k**2)`` — matching the per-(packet, subcarrier, antenna) sampling
    granularity that T3.3 modeled — and rotates the complex CSI by
    ``exp(j*phi)``.

    Magnitude ``|H|`` is preserved exactly; only phase changes. This is the
    physical correctness guarantee of the augmentation: a real chipset swap
    would shift phase, not magnitude. (And it is precisely *why* this slice
    needed the real-imag stacked loader — a magnitude-only encoder would be
    blind to the augmentation.)
    """
    mu_np, sigma_np = load_profile(Path(profile_path))
    mu_buf = torch.from_numpy(mu_np).float()  # (S,)
    sigma_buf = torch.from_numpy(sigma_np).float()  # (S,)

    def _aug(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"phase-noise aug expects (B, T, S, 2*A); got shape {tuple(x.shape)}"
            )
        b, t, s, double_a = x.shape
        if double_a % 2 != 0:
            raise ValueError(
                f"last dim must be 2*A (real-imag stacked); got {double_a}"
            )
        a = double_a // 2

        # Split real-imag back into a complex view of CSI.
        re = x[..., :a]
        im = x[..., a:]
        csi = torch.complex(re, im)  # (B, T, S, A) complex64 (since re/im are f32)

        # Move profile to the input device the first time we see one and
        # broadcast over (B, T, A). Each call resamples its own randomness so
        # SimCLR's two views differ.
        mu = mu_buf.to(x.device).view(1, 1, s, 1)
        sigma = sigma_buf.to(x.device).view(1, 1, s, 1)
        phi = torch.randn(b, t, s, a, device=x.device, dtype=x.dtype) * sigma + mu

        rot = torch.complex(torch.cos(phi), torch.sin(phi))
        rotated = csi * rot

        return torch.cat([rotated.real, rotated.imag], dim=-1).to(x.dtype)

    return _aug
