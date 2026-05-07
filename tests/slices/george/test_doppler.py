"""T1.5 sanity test for `doppler_warp`.

A pure-tone CSI of frequency f0 stretched by factor 2.0 (then re-cropped to
the original window length) should expose the *first half* of the stretched
signal — half a wavelength's worth of cycles — so its dominant frequency in
the cropped FFT lands near f0/2.
"""

from __future__ import annotations

import math

import torch

from src.slices.george.augmentations import doppler_warp


def test_doppler_factor_two_halves_dominant_frequency():
    t_steps = 256
    f0_cycles = 8.0  # cycles over the 256-sample window

    t = torch.linspace(0, 1, t_steps)
    x = torch.sin(2 * math.pi * f0_cycles * t).reshape(t_steps, 1, 1)

    y = doppler_warp(x, factor=2.0)

    fft_x = torch.fft.rfft(x[:, 0, 0]).abs()
    fft_y = torch.fft.rfft(y[:, 0, 0]).abs()
    peak_x = int(fft_x.argmax())
    peak_y = int(fft_y.argmax())

    assert peak_x == int(
        round(f0_cycles)
    ), f"input dominant bin should be {int(round(f0_cycles))}, got {peak_x}"
    assert abs(peak_y - peak_x // 2) <= 2, (
        f"after factor=2.0 stretch+crop, dominant frequency should "
        f"be near {peak_x // 2}, got {peak_y}"
    )


def test_doppler_random_factor_preserves_shape():
    sample = torch.randn(100, 30, 3)
    out = doppler_warp(sample)
    assert out.shape == sample.shape


def test_doppler_batched_independent_factors():
    batch = torch.randn(4, 100, 30, 3)
    out = doppler_warp(batch)
    assert out.shape == batch.shape
