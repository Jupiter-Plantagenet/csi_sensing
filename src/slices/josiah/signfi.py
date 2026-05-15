"""SignFi loader — UL/DL paired CSI for CAPC dual-view reproduction.

Dataset: SignFi (Ma et al. 2018, ACM IMWUT). Sign-language recognition with
276 sign-word classes collected by the Linux 802.11n CSI Tool. Two
environments:

* Lab: ``dataset_lab_276_dl.mat`` (DL, 5520 instances) + ``dataset_lab_276_ul.mat``
  (UL, 5520 instances). Pre-training environment in CAPC §4.1.1.
* Home: ``dataset_home_276.mat`` containing both ``csid_home`` (DL) and
  ``csiu_home`` (UL), 2760 instances total. Downstream eval environment in
  CAPC Table 1.

CSI shape per environment: ``(200, 30, 3, N_instances)`` complex128 —
200 packets × 30 subcarriers × 3 antennas. Labels are 1..276 (uint16).

CAPC consumes paired ``(view_a=DL, view_b=UL)`` tensors per sample,
reshaped to ``(Na=3, Ns=30, Nt=200)`` then split into ``L=20`` windows of
``(3, 30, 10)`` so the GRU autoregressor has 20 timesteps. We hand the
two-stream CAPC model ``(B, L, Na, Ns, Nt)``-shaped views.

Amplitude only by default (CAPC §2.1: "amplitude being more stable").
Min-max normalized globally over training-data amplitude (matches the
paper's per-dataset normalization).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

NUM_SIGNFI_CLASSES = 276
SIGNFI_NT = 200
SIGNFI_NS = 30
SIGNFI_NA = 3

SignFiEnvironment = Literal["home", "lab"]


def _load_mat_paired(
    env: SignFiEnvironment, root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(dl, ul, labels)`` for the requested SignFi environment.

    ``dl`` and ``ul`` are complex128 arrays of shape ``(N, 3, 30, 200)``
    (already transposed from the on-disk ``(200, 30, 3, N)``). ``labels``
    is int64 of shape ``(N,)`` with values in ``[0, 275]``.
    """
    if env == "home":
        path = root / "dataset_home_276.mat"
        mat = sio.loadmat(str(path))
        dl = mat["csid_home"]
        ul = mat["csiu_home"]
        labels = mat["label_home"].ravel()
    elif env == "lab":
        dl_path = root / "dataset_lab_276_dl.mat"
        ul_path = root / "dataset_lab_276_ul.mat"
        dl_mat = sio.loadmat(str(dl_path))
        ul_mat = sio.loadmat(str(ul_path))
        # The Lab .mat files use different variable names per the SignFi
        # README example ("csid_lab", "label_lab"); we support both
        # conventions.
        dl = dl_mat.get("csid_lab", dl_mat.get("csi"))
        ul = ul_mat.get("csiu_lab", ul_mat.get("csi"))
        labels = (
            dl_mat.get("label_lab", dl_mat.get("label")).ravel()
            if "label_lab" in dl_mat or "label" in dl_mat
            else ul_mat.get("label_lab", ul_mat.get("label")).ravel()
        )
    else:
        raise ValueError(f"unknown SignFi environment: {env!r}")

    # On-disk: (200, 30, 3, N). Transpose to (N, 3, 30, 200) for batching.
    dl = np.transpose(dl, (3, 2, 1, 0)).astype(np.complex64)
    ul = np.transpose(ul, (3, 2, 1, 0)).astype(np.complex64)
    labels = labels.astype(np.int64) - 1  # 1..276 -> 0..275
    return dl, ul, labels


def _to_windows(x: torch.Tensor, num_windows: int = 20) -> torch.Tensor:
    """``(B, Na, Ns, Nt)`` -> ``(B, L, Na, Ns, Nt/L)`` for the CAPC GRU."""
    b, na, ns, nt = x.shape
    if nt % num_windows != 0:
        raise ValueError(f"Nt={nt} not divisible by L={num_windows}")
    nt_per_window = nt // num_windows
    return x.reshape(b, na, ns, num_windows, nt_per_window).permute(0, 3, 1, 2, 4).contiguous()


class SignFiPaired(Dataset):
    """SignFi UL/DL paired dataset for CAPC dual-view SSL.

    Each ``__getitem__`` returns ``(view_a=DL, view_b=UL, label)`` where the
    two views are shaped ``(L=20, Na=3, Ns=30, Nt_per_window=10)`` amplitude
    tensors. Normalization is min-max over the training partition (the
    statistics are computed on first construction and cached so the
    val/test partitions reuse them).
    """

    def __init__(
        self,
        *,
        env: SignFiEnvironment,
        root: str | Path = "data",
        indices: list[int] | None = None,
        normalize_stats: tuple[float, float] | None = None,
        num_windows: int = 20,
        cache_path: str | Path | None = None,
    ) -> None:
        root_p = Path(root)
        cache_meta = {
            "env": env,
            "num_windows": num_windows,
            "format": "signfi-paired-amplitude-minmax-v1",
        }
        self.meta = cache_meta
        self.num_windows = num_windows

        if cache_path is not None and Path(cache_path).exists():
            blob = torch.load(cache_path, weights_only=False)
            if blob.get("meta") != cache_meta:
                raise ValueError(
                    f"cache at {cache_path} built with {blob.get('meta')!r}, "
                    f"loader requested {cache_meta!r}"
                )
            self._dl = blob["dl"]
            self._ul = blob["ul"]
            self._y = blob["y"]
            self.norm_min = float(blob["norm_min"])
            self.norm_max = float(blob["norm_max"])
        else:
            dl_c, ul_c, y = _load_mat_paired(env, root_p)
            dl_amp = np.abs(dl_c).astype(np.float32)  # (N, 3, 30, 200)
            ul_amp = np.abs(ul_c).astype(np.float32)
            if normalize_stats is None:
                vmin = float(min(dl_amp.min(), ul_amp.min()))
                vmax = float(max(dl_amp.max(), ul_amp.max()))
            else:
                vmin, vmax = normalize_stats
            scale = max(vmax - vmin, 1e-6)
            dl_amp = (dl_amp - vmin) / scale
            ul_amp = (ul_amp - vmin) / scale
            self._dl = torch.from_numpy(dl_amp)
            self._ul = torch.from_numpy(ul_amp)
            self._y = torch.from_numpy(y)
            self.norm_min = vmin
            self.norm_max = vmax
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "dl": self._dl,
                        "ul": self._ul,
                        "y": self._y,
                        "norm_min": vmin,
                        "norm_max": vmax,
                        "meta": cache_meta,
                    },
                    cache_path,
                )

        if indices is None:
            self._index = np.arange(len(self._y))
        else:
            self._index = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = int(self._index[i])
        dl = self._dl[idx]  # (3, 30, 200)
        ul = self._ul[idx]
        dl_w = _to_windows(dl.unsqueeze(0))[0]  # (20, 3, 30, 10)
        ul_w = _to_windows(ul.unsqueeze(0))[0]
        return dl_w, ul_w, int(self._y[idx].item())


def k_shot_split(
    labels: np.ndarray, k: int, *, num_classes: int = NUM_SIGNFI_CLASSES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class ``k``-shot training indices + remainder as test indices.

    Mirrors CAPC Table 1 evaluation: pick ``k`` labelled instances per class
    for the linear-probe training set; remaining instances become the test
    set. Returns ``(train_idx, test_idx)``.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)
        k_eff = min(k, cls_idx.size)
        train_idx.extend(cls_idx[:k_eff].tolist())
        test_idx.extend(cls_idx[k_eff:].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)
