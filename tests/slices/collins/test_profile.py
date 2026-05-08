"""T3.4 — Phase-noise profile sanity test.

Synthesise CSI with a known per-(packet, subcarrier, antenna) Gaussian phase
distribution, inject the noise into a constant clean channel, run the T3.3
extraction and fit, and check the recovered (mu, sigma) match the truth. The
test reports the per-subcarrier KS statistic against N(0, sigma_truth) per the
issue contract (#53).

Self-contained: no real Widar3.0 data on disk is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from src.slices.collins.phase_profile import (
    extract_phase_residuals,
    fit_per_subcarrier_gaussian,
)

# Synthesis parameters. Sigma kept small (0.3 rad ~= 17 deg) so the Gaussian
# stays well clear of the +-pi wrap; this is the regime the contract default
# fits cleanly. Larger sigmas would drift toward the uniform-on-circle case,
# which the per-subcarrier-Gaussian model is not designed to capture.
_T = 1024
_S = 30
_A = 3
_SIGMA_TRUTH = 0.3
_SEED = 42

# Tolerances. With T*A = 3072 samples per subcarrier the standard error of a
# Gaussian-fit sigma is ~ sigma_truth / sqrt(2 * (T*A - 1)) ~= 0.004; we leave
# headroom (5x) for cross-subcarrier worst-case before declaring failure.
_MU_TOL = 0.02
_SIGMA_TOL = 0.02

# KS critical value at alpha=0.05 for n=3072 is ~ 1.36/sqrt(n) ~= 0.0245. Use
# 0.05 to keep the test stable across reasonable seeds; true-Gaussian data
# clears this comfortably.
_KS_TOL = 0.05


def _synthesise_gaussian_phase_csi(
    sigma_truth: float, seed: int = _SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Build (H, phi) where H[t, k, a] = H_clean[k, a] * exp(j * phi[t, k, a]).

    phi is i.i.d. N(0, sigma_truth^2). H_clean is a random non-degenerate
    complex channel (so |H_bar| stays well above the dead-subcarrier guard).
    Returns (H, phi) with phi for ground-truth comparison if needed.
    """
    rng = np.random.default_rng(seed)
    h_clean = rng.standard_normal(size=(_S, _A)) + 1j * rng.standard_normal(
        size=(_S, _A)
    )
    h_clean = h_clean.astype(np.complex128)
    phi = rng.normal(loc=0.0, scale=sigma_truth, size=(_T, _S, _A))
    H = h_clean[None, :, :] * np.exp(1j * phi)
    return H, phi


def test_profile_recovers_known_gaussian_per_subcarrier(capsys):
    """fit_per_subcarrier_gaussian recovers the synthesised (mu, sigma)."""
    H, _phi = _synthesise_gaussian_phase_csi(_SIGMA_TRUTH)

    residuals = extract_phase_residuals(H)  # (T, S, A) float
    assert residuals.shape == (_T, _S, _A)

    # Reshape so each row is one (packet, antenna) sample, columns are subcarriers.
    flat = residuals.transpose(0, 2, 1).reshape(-1, _S)  # (T*A, S)
    assert flat.shape == (_T * _A, _S)

    mu_hat, sigma_hat = fit_per_subcarrier_gaussian(flat)

    # Per-subcarrier KS statistic against the truth distribution.
    ks_stats = np.array(
        [
            kstest(flat[:, k], "norm", args=(0.0, _SIGMA_TRUTH)).statistic
            for k in range(_S)
        ]
    )

    # Report (visible with `pytest -s`).
    with capsys.disabled():
        print(f"\n[T3.4] mu_hat:    min={mu_hat.min():+.5f}  max={mu_hat.max():+.5f}")
        print(
            f"[T3.4] sigma_hat: min={sigma_hat.min():.5f}  max={sigma_hat.max():.5f}"
            f"  (truth={_SIGMA_TRUTH})"
        )
        print(
            f"[T3.4] KS stats vs N(0, {_SIGMA_TRUTH}^2): "
            f"min={ks_stats.min():.4f}  median={float(np.median(ks_stats)):.4f}"
            f"  max={ks_stats.max():.4f}  (tol < {_KS_TOL})"
        )

    # Moment matching (the contract's "or" alternative).
    np.testing.assert_allclose(mu_hat, 0.0, atol=_MU_TOL)
    np.testing.assert_allclose(sigma_hat, _SIGMA_TRUTH, atol=_SIGMA_TOL)

    # KS test (the contract's primary).
    assert ks_stats.max() < _KS_TOL, (
        f"max KS statistic {ks_stats.max():.4f} exceeds tolerance {_KS_TOL}; "
        f"the recovered residual distribution is no longer "
        f"indistinguishable from N(0, {_SIGMA_TRUTH}^2) at this sample size."
    )


def test_extract_phase_residuals_shape_invariants():
    """Sanity: residuals have the same shape as input and live in (-pi, pi]."""
    H, _phi = _synthesise_gaussian_phase_csi(_SIGMA_TRUTH)
    residuals = extract_phase_residuals(H)
    assert residuals.shape == H.shape
    assert residuals.dtype.kind == "f"
    assert residuals.min() > -np.pi - 1e-9
    assert residuals.max() <= np.pi + 1e-9


def test_extract_phase_residuals_rejects_2d_input():
    """The (T, S, A) shape is part of the API contract."""
    bad = np.zeros((10, 30), dtype=np.complex128)
    with pytest.raises(ValueError, match="expected"):
        extract_phase_residuals(bad)
