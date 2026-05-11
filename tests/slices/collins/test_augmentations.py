"""T3.5 — augmentations.py tests.

Unit tests for the generic baseline and phase-noise injection augmentations
introduced in #54. The interesting invariants:

- shape is preserved (same `(B, T, S, 2*A)` in and out);
- magnitude is preserved by phase-noise injection (it's a unitary rotation);
- two consecutive calls give different results (each call resamples its
  randomness, so SimCLR's two views differ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.slices.collins.augmentations import (
    make_generic_aug,
    make_phase_noise_aug,
)
from src.slices.collins.phase_profile import save_profile


def _toy_csi_batch(b: int = 2, t: int = 64, s: int = 30, a: int = 3) -> torch.Tensor:
    """Random real-imag-stacked CSI batch shaped ``(B, T, S, 2*A)``."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(b, t, s, 2 * a, generator=g, dtype=torch.float32)


def test_generic_aug_preserves_shape_and_is_stochastic():
    aug = make_generic_aug(sigma=0.3, subcarrier_drop_prob=0.15)
    x = _toy_csi_batch()
    y1 = aug(x)
    y2 = aug(x)
    assert y1.shape == x.shape
    assert y2.shape == x.shape
    # Different stochastic draws => the two views must differ.
    assert not torch.equal(y1, y2)
    # Generic aug shifts and masks; output should differ from input as well.
    assert not torch.equal(y1, x)


def test_phase_noise_aug_preserves_shape_and_magnitude(tmp_path: Path):
    # Build a tiny synthetic profile (mu=0, sigma=0.5) and save it like T3.3 would.
    s = 30
    profile_path = tmp_path / "phase_profile.npz"
    save_profile(
        profile_path,
        mu=np.zeros(s, dtype=np.float32),
        sigma=np.full(s, 0.5, dtype=np.float32),
        num_samples_per_subcarrier=1024,
        meta={"data_root": "synthetic", "users": [], "num_files": 0, "seed": 0},
    )

    aug = make_phase_noise_aug(profile_path)
    x = _toy_csi_batch(b=2, t=64, s=s, a=3)

    y = aug(x)

    # Shape preserved
    assert y.shape == x.shape

    # Magnitude preserved per (b, t, s, a). Because the augmentation rotates
    # the complex CSI by exp(j*phi) — a unitary operation — sqrt(re**2+im**2)
    # for each antenna must match the input within float-precision tolerance.
    a = x.shape[-1] // 2
    re_in, im_in = x[..., :a], x[..., a:]
    re_out, im_out = y[..., :a], y[..., a:]
    mag_in = torch.sqrt(re_in**2 + im_in**2)
    mag_out = torch.sqrt(re_out**2 + im_out**2)
    torch.testing.assert_close(mag_in, mag_out, rtol=1e-5, atol=1e-5)

    # And — confirming the augmentation is non-trivial — the real/imag parts
    # do change when phase rotates.
    assert not torch.equal(y, x)


def test_phase_noise_aug_is_stochastic(tmp_path: Path):
    s = 30
    profile_path = tmp_path / "phase_profile.npz"
    save_profile(
        profile_path,
        mu=np.zeros(s, dtype=np.float32),
        sigma=np.full(s, 0.5, dtype=np.float32),
        num_samples_per_subcarrier=1024,
        meta={"data_root": "synthetic", "users": [], "num_files": 0, "seed": 0},
    )
    aug = make_phase_noise_aug(profile_path)
    x = _toy_csi_batch()
    y1 = aug(x)
    y2 = aug(x)
    # Each call resamples; SimCLR's two views must differ.
    assert not torch.equal(y1, y2)


def test_phase_noise_aug_rejects_odd_last_dim(tmp_path: Path):
    s = 30
    profile_path = tmp_path / "phase_profile.npz"
    save_profile(
        profile_path,
        mu=np.zeros(s, dtype=np.float32),
        sigma=np.full(s, 0.5, dtype=np.float32),
        num_samples_per_subcarrier=1024,
        meta={"data_root": "synthetic", "users": [], "num_files": 0, "seed": 0},
    )
    aug = make_phase_noise_aug(profile_path)
    bad = torch.zeros(2, 64, s, 5)  # last dim must be 2*A; 5 is odd
    try:
        aug(bad)
    except ValueError as e:
        assert "2*A" in str(e) or "real-imag" in str(e)
    else:
        raise AssertionError("expected ValueError on odd last dim")
