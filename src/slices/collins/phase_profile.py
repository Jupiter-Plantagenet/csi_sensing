"""T3.3 — Per-chipset phase-noise profile fit.

Walks the train half of the Widar3.0 cross-subject split, extracts phase
residuals from each gesture instance, and fits an **independent Gaussian per
subcarrier** — the default model from #52. The result is saved as a small
``.npz`` so T3.4 can sanity-test it and T3.5 can sample from it.

Channel-response removal
------------------------
For one .dat file (one gesture × receiver), CSI has shape ``(T, S=30, A=3)``
complex. The "channel response" is the slowly-varying component of CSI within
that 1–2 s window — geometry of the room, person's slow motion. We approximate
it with the per-(subcarrier, antenna) **temporal mean** of complex CSI:

    H_bar[k, a] = mean over t of  H[t, k, a]

Phase residual is what is left after dividing each packet by this mean:

    phi_res[t, k, a] = arg( H[t, k, a] / H_bar[k, a] )    in (-pi, pi]

Complex division before ``arg`` handles 2-pi wrapping cleanly: if the channel
phase happens to sit near ±pi, naive subtraction would wrap-bias the residual.

Why a per-subcarrier Gaussian?
------------------------------
Halperin et al. 2011 and follow-ups characterise the Intel 5300 NIC's phase
distortions as a sum of per-packet linear-in-subcarrier terms (CFO, STO, PDD)
plus per-packet random offsets. A finer model would fit these terms jointly;
the issue-#52 default is the simplest model that captures the *amplitude* of
the residual phase noise on each subcarrier, which is the part the augmentation
needs to sample from. T3.4 will sanity-check this model; if it fails, Stage B
on CSI-Bench is the natural place to upgrade.

Empirical observation (Stage A, this fit on Widar3.0 train)
-----------------------------------------------------------
The fitted ``sigma`` values land at ~1.80 rad uniformly across all 30
subcarriers. The std of a uniform distribution on (-pi, pi] is pi/sqrt(3) ~=
1.814; ours sits just below that. The interpretation is that the temporal
mean is too coarse a channel estimate over a 1-2 s gesture window — within
that window the channel itself drifts (person motion, Doppler), so
"residual" is motion-plus-hardware-noise rather than hardware-noise alone.

This is fine for the slice's tracer-bullet path: T3.5 still has a sampler,
T3.6 still has a comparison. The slice's real causal claim is Stage B
(cross-chipset on CSI-Bench), where chipset-specific shifts dominate the
residual and the per-subcarrier Gaussian becomes physically meaningful. A
refined Stage A model — short sliding-window mean, or per-packet linear
phase removal a la Halperin — is straightforward future work; we ship the
contract default first.

Saved schema (``phase_profile.npz``)
------------------------------------
- ``mu``    : (S=30,)  float32  per-subcarrier residual mean (radians)
- ``sigma`` : (S=30,)  float32  per-subcarrier residual std  (radians)
- ``num_samples_per_subcarrier`` : int64  total (packet * antenna * file) used
- ``meta``  : 0-d object array carrying the run config dict
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import DEFAULT_TRAIN_USERS, NUM_CLASSES, _FILENAME_RE

CSI_S = 30  # Intel 5300 subcarriers
_EPS = 1e-9


def extract_phase_residuals(csi: np.ndarray) -> np.ndarray:
    """Per-packet phase residual after channel-response removal.

    Parameters
    ----------
    csi : (T, S, A) complex
        Raw CSI for one gesture instance × receiver.

    Returns
    -------
    (T, S, A) float in (-pi, pi]
    """
    if csi.ndim != 3:
        raise ValueError(f"expected (T, S, A) complex; got shape {csi.shape}")
    h_bar = csi.mean(axis=0, keepdims=True)  # (1, S, A) complex
    # Guard against dead subcarrier/antenna pairs whose mean is ~0; if any
    # show up, leave that residual at 0 rather than dividing by ~0.
    safe = np.where(np.abs(h_bar) < _EPS, 1.0 + 0.0j, h_bar)
    return np.angle(csi / safe)


def fit_per_subcarrier_gaussian(
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-subcarrier mean and std of a stack of phase residuals.

    Parameters
    ----------
    residuals : (N, S) float

    Returns
    -------
    mu, sigma : (S,) float32 each
    """
    if residuals.ndim != 2:
        raise ValueError(f"expected (N, S); got shape {residuals.shape}")
    mu = residuals.mean(axis=0).astype(np.float32)
    sigma = residuals.std(axis=0).astype(np.float32)
    return mu, sigma


def list_train_dat_files(
    root: Path, users: tuple[int, ...] = DEFAULT_TRAIN_USERS
) -> list[Path]:
    """All Widar3.0 .dat files under ``root`` whose user is in the train split."""
    user_set = set(users)
    paths: list[Path] = []
    for dat in sorted(root.rglob("*.dat")):
        m = _FILENAME_RE.match(dat.name)
        if m is None:
            continue
        u = int(m.group(1))
        g = int(m.group(2))
        if u in user_set and 1 <= g <= NUM_CLASSES:
            paths.append(dat)
    return paths


def _parse_dat_complex(path: Path) -> np.ndarray:
    """Parse one .dat into (T, 30, 3) complex128. Lazy-imports csiread."""
    import csiread

    c = csiread.Intel(str(path), nrxnum=3, ntxnum=1)
    c.read()
    return np.asarray(c.csi).squeeze(-1)


def fit_profile(
    data_root: Path,
    users: tuple[int, ...] = DEFAULT_TRAIN_USERS,
    num_files: int | None = 500,
    seed: int = 0,
    min_packets: int = 100,
    progress_every: int = 50,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Walk train data, accumulate phase-residual statistics, return the fit.

    Returns ``(mu, sigma, files_used, samples_per_subcarrier)``.

    The accumulator uses streaming sufficient statistics (sum, sum of squares)
    so memory is O(S), not O(N samples). Numerical drift is negligible at the
    sample sizes we use here (~10^7).
    """
    files = list_train_dat_files(data_root, users)
    if not files:
        raise RuntimeError(
            f"phase_profile.fit_profile: no .dat files found under {data_root!s} "
            f"for users {sorted(users)}. Has CSI_*.zip been extracted?"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    if num_files is not None:
        files = files[:num_files]

    sum_x = np.zeros(CSI_S, dtype=np.float64)
    sum_x2 = np.zeros(CSI_S, dtype=np.float64)
    count = 0  # all subcarriers see the same number of samples
    files_used = 0

    for i, path in enumerate(files):
        try:
            csi = _parse_dat_complex(path)
        except Exception as exc:
            print(f"  skip {path.name}: parse error: {exc}")
            continue
        if csi.shape[0] < min_packets:
            continue
        if csi.shape[1] != CSI_S:
            print(f"  skip {path.name}: unexpected subcarrier count {csi.shape[1]}")
            continue

        res = extract_phase_residuals(csi)  # (T, S, A) float
        # Reshape to (T*A, S) so each row is one (packet, antenna) sample.
        flat = res.transpose(0, 2, 1).reshape(-1, CSI_S)
        sum_x += flat.sum(axis=0)
        sum_x2 += (flat**2).sum(axis=0)
        count += flat.shape[0]
        files_used += 1

        if (i + 1) % progress_every == 0:
            print(f"  processed {i + 1}/{len(files)} files (used {files_used})")

    if count == 0:
        raise RuntimeError(
            "phase_profile.fit_profile: every candidate file was skipped; "
            "check parse errors and min_packets."
        )

    mu = (sum_x / count).astype(np.float32)
    var = sum_x2 / count - (sum_x / count) ** 2
    sigma = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

    return mu, sigma, files_used, count


def save_profile(
    path: Path,
    mu: np.ndarray,
    sigma: np.ndarray,
    num_samples_per_subcarrier: int,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu=mu,
        sigma=sigma,
        num_samples_per_subcarrier=np.int64(num_samples_per_subcarrier),
        meta=np.array(meta, dtype=object),
    )


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (mu, sigma) from a saved profile. Used by T3.4 / T3.5."""
    blob = np.load(path, allow_pickle=True)
    return blob["mu"], blob["sigma"]


def main() -> None:
    parser = argparse.ArgumentParser(description="T3.3 phase-noise profile fit")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/widar3/extracted",
        help="extracted Widar3.0 root (parent of YYYYMMDD/userN/*.dat)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/widar3/cache/phase_profile.npz",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=500,
        help="cap the number of .dat files used for the fit",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)

    print(f"[T3.3] fitting phase-noise profile from {data_root}")
    mu, sigma, files_used, n = fit_profile(
        data_root,
        users=DEFAULT_TRAIN_USERS,
        num_files=args.num_files,
        seed=args.seed,
    )

    print(f"[T3.3] files used: {files_used}")
    print(f"[T3.3] samples per subcarrier: {n:,}")
    print(f"[T3.3] mu range:    [{mu.min():+.4f}, {mu.max():+.4f}] rad")
    print(f"[T3.3] sigma range: [{sigma.min():+.4f}, {sigma.max():+.4f}] rad")

    save_profile(
        output,
        mu,
        sigma,
        n,
        {
            "data_root": str(data_root),
            "users": list(DEFAULT_TRAIN_USERS),
            "num_files": files_used,
            "seed": args.seed,
        },
    )
    print(f"[T3.3] saved: {output}")


if __name__ == "__main__":
    main()
