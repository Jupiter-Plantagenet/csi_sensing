"""UT-HAR loader for published-baseline reproductions.

Dataset: UT-HAR (Yousefi et al. 2017, reorganized by SenseFi).
7 activity classes (lie down, fall, walk, run, sit down, stand up, pick up).
Pre-split into ``X_{train,val,test}.csv`` (actually ``.npy``) and matching
labels under ``data/ut_har/UT_HAR/``.

Per the SenseFi `dataset.py::UT_HAR_dataset`:

* Each ``X_<split>.csv`` is a ``.npy`` array of shape ``(N, 250, 90)``.
* Reshape per-sample to ``(1, 250, 90)`` (1 channel × 250 packets × 90 features).
* Normalize per-file: ``(x - x.min()) / (x.max() - x.min())``.

Used by AutoFi §IV-C (UT-HAR 20-shot = 0.788) and SSLCSI Table 4c
(MAE on UT-HAR = 0.843).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

UT_HAR_T = 250
UT_HAR_F = 90
NUM_UT_HAR_CLASSES = 7


def _load_npy(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.load(f)


def load_ut_har_split(
    split: str, *, root: str | Path = "data/ut_har/UT_HAR"
) -> tuple[np.ndarray, np.ndarray]:
    """Load one UT-HAR split. Returns ``(X[N, 1, 250, 90], y[N])``."""
    root_p = Path(root)
    x = _load_npy(root_p / "data" / f"X_{split}.csv").astype(np.float32)
    y = _load_npy(root_p / "label" / f"y_{split}.csv").astype(np.int64)
    # SenseFi normalization is per-file global min-max.
    vmin = float(x.min())
    vmax = float(x.max())
    x = (x - vmin) / max(vmax - vmin, 1e-6)
    # SenseFi shape: (N, 1, 250, 90) - one channel as the "image" axis.
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x, y


class UTHARDataset(Dataset):
    """UT-HAR split with optional ``indices`` subset.

    ``__getitem__`` returns ``(x[1, 250, 90], y)``.
    """

    def __init__(
        self,
        split: str,
        *,
        root: str | Path = "data/ut_har/UT_HAR",
        indices: list[int] | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        cache_meta = {"split": split, "format": "ut-har-1x250x90-minmax-v1"}
        self.meta = cache_meta
        if cache_path is not None and Path(cache_path).exists():
            blob = torch.load(cache_path, weights_only=False)
            if blob.get("meta") != cache_meta:
                raise ValueError(
                    f"cache at {cache_path} built with {blob.get('meta')!r}"
                )
            self._x = blob["x"]
            self._y = blob["y"]
        else:
            x_np, y_np = load_ut_har_split(split, root=root)
            self._x = torch.from_numpy(x_np)
            self._y = torch.from_numpy(y_np)
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"x": self._x, "y": self._y, "meta": cache_meta}, cache_path)
        if indices is None:
            self._index = np.arange(len(self._y))
        else:
            self._index = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self._index[i])
        return self._x[idx], int(self._y[idx].item())


def ut_har_k_shot_indices(
    labels: np.ndarray, k: int, *, num_classes: int = NUM_UT_HAR_CLASSES, seed: int = 42
) -> np.ndarray:
    """Return ``k`` random indices per class. Used by AutoFi §IV-C FSC."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[: min(k, cls_idx.size)].tolist())
    return np.asarray(chosen, dtype=np.int64)
