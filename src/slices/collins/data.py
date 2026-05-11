"""Dataset wrappers for Slice 3.

T3.1 shipped `StubCSI` so the pipeline could run before any real data was
wired in. T3.2 (this file) adds `Widar3CrossSubject`, the real Widar3.0 raw
CSI loader for the cross-subject split mandated by issue #51.

Stub sample shape is (1024, 30, 3) per the T3.1 issue (#50): 1024 packets at
~1 kHz approximates a one-second gesture window; 30 subcarriers and 3 antenna
pairs match the Intel 5300 NIC layout described in docs/slice-1-afk-plan.md
section 6.

Real-data convention (`Widar3CrossSubject`): each sample is a CSI tensor
shaped (T=1024, S=30, A=6) of float32, where the last axis stacks the real
parts of the three RX antennas followed by their imaginary parts. Phase
information is preserved because phase-noise injection (T3.5) operates by
multiplying the complex CSI by exp(j·phi); a magnitude-only encoder would be
blind to that augmentation.

Filename convention used by Widar3.0 .dat files (verified against
CSI_20181128.zip during T3.2 reconnaissance):

    user<U>-<G>-<T>-<O>-<I>-r<R>.dat

where U=user 1..17, G=gesture 1..6, T=torso location 1..5, O=face
orientation 1..5, I=instance 1..5, R=receiver 1..6.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

CSI_T = 1024
CSI_S = 30
CSI_A = 3
NUM_CLASSES = 6

# Real-data antenna axis after real/imag stacking.
CSI_A_REAL = 2 * CSI_A

_FILENAME_RE = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat$")

DEFAULT_TRAIN_USERS: tuple[int, ...] = tuple(range(1, 14))  # users 1..13
DEFAULT_TEST_USERS: tuple[int, ...] = (14, 15, 16, 17)


class StubCSI(Dataset):
    """Random CSI samples — pipeline plumbing only, no real signal."""

    def __init__(
        self,
        num_samples: int = 16,
        time_steps: int = CSI_T,
        subcarriers: int = CSI_S,
        antenna_pairs: int = CSI_A,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self._x = torch.randn(
            num_samples, time_steps, subcarriers, antenna_pairs, generator=g
        )
        self._y = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x[i], int(self._y[i].item())


class Widar3CrossSubject(Dataset):
    """Widar3.0 raw-CSI loader with the cross-subject split required by #51.

    Walks ``<root>/<YYYYMMDD>/user<U>/user<U>-<G>-<T>-<O>-<I>-r<R>.dat`` files,
    keeps only those whose user matches the train or test split, and parses
    each .dat lazily via csiread.Intel. Each .dat — i.e. each (gesture
    instance × receiver) tuple — becomes one sample.

    A small disk cache (``<cache_dir>/<day>__<basename>.npy``) stores the
    cropped, normalized real-imag-stacked tensor on first read. Subsequent
    reads bypass csiread entirely; the cache is the bottleneck-free fast path
    for multi-epoch pre-training.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        train_users: Iterable[int] = DEFAULT_TRAIN_USERS,
        test_users: Iterable[int] = DEFAULT_TEST_USERS,
        time_steps: int = CSI_T,
        num_classes: int = NUM_CLASSES,
        cache_dir: str | Path | None = None,
        max_files: int | None = None,
        shuffle_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.users: set[int] = set(train_users) if train else set(test_users)
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.files: list[tuple[Path, int, int]] = []  # (path, user_id, label)
        for dat in sorted(self.root.rglob("*.dat")):
            m = _FILENAME_RE.match(dat.name)
            if m is None:
                continue
            user_id = int(m.group(1))
            gesture = int(m.group(2))
            if user_id in self.users and 1 <= gesture <= self.num_classes:
                self.files.append((dat, user_id, gesture - 1))

        # Deterministic shuffle so a small `max_files` subset still spans all
        # gesture classes — alphabetical order would land all gesture-1 files
        # first and leave the linear probe with only one class.
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(self.files)

        if max_files is not None:
            self.files = self.files[:max_files]

        if not self.files:
            raise RuntimeError(
                f"Widar3CrossSubject({self.root!s}, train={train}) found 0 .dat "
                f"files matching users {sorted(self.users)}. Has CSI_*.zip been "
                f"extracted under {self.root}/?"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, _user, label = self.files[i]
        x = self._load_and_process(path)
        return torch.from_numpy(x), label

    # --------------- internals ---------------

    def _cache_path(self, path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        # day folder is the parent of `user<U>/`, i.e. the grandparent of the .dat
        day = path.parent.parent.name
        return self.cache_dir / f"{day}__{path.name}.npy"

    def _load_and_process(self, path: Path) -> np.ndarray:
        cache_path = self._cache_path(path)
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path)

        import csiread  # local import: keeps test_smoke.py importable on CI

        c = csiread.Intel(str(path), nrxnum=3, ntxnum=1)
        c.read()
        # csiread.Intel.csi has shape (T, S, Nrx, Ntx); Ntx=1 here.
        csi = np.asarray(c.csi).squeeze(-1)  # (T, 30, 3) complex128
        csi = self._crop_or_pad_time(csi, self.time_steps)

        # Real-imag stack along the antenna axis: (T, 30, 3) complex
        # -> (T, 30, 6) float32 with channels [Re_a1, Re_a2, Re_a3, Im_a1, Im_a2, Im_a3].
        re = csi.real.astype(np.float32)
        im = csi.imag.astype(np.float32)
        x = np.concatenate([re, im], axis=-1)

        # Per-sample z-score. Magnitudes from get_scaled_csi() are in linear units
        # spanning ~1 to ~50; raw real/imag parts can be negative. Z-score gives
        # the encoder a stable input distribution without losing relative phase.
        mu = x.mean()
        sigma = x.std() + 1e-8
        x = (x - mu) / sigma

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, x)

        return x

    @staticmethod
    def _crop_or_pad_time(csi: np.ndarray, time_steps: int) -> np.ndarray:
        T = csi.shape[0]
        if T == time_steps:
            return csi
        if T > time_steps:
            start = (T - time_steps) // 2
            return csi[start : start + time_steps]
        pad_shape = (time_steps - T,) + csi.shape[1:]
        pad = np.zeros(pad_shape, dtype=csi.dtype)
        return np.concatenate([csi, pad], axis=0)
