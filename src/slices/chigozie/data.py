"""Dataset wrappers for Slice 2 (static-component perturbation).

T2.1 shipped `StubCSI`. T2.2 adds `Widar3CrossEnvironment` for the real
cross-environment evaluation that this slice's augmentation targets.

Convention: each sample is a real-valued CSI tensor shaped `(T, S, A)`:
    T = time samples (cropped/padded to a fixed length)
    S = subcarriers (30 for Intel 5300)
    A = antenna pairs (3 for the standard 1-TX × 3-RX Widar3.0 setup)
plus a single integer gesture label in `[0, NUM_CLASSES)`.

The Widar3.0 raw release ships `.dat` files in the Linux 802.11n CSI Tool
format. Parsed via `csiread.Intel(nrxnum=3, ntxnum=1)`; complex CSI is
projected to its magnitude as the real-valued tensor consumed by the
encoder.

Cross-environment vs cross-subject:
- Cross-subject splits by user ID (Slices 1, 4, 5, 6 default).
- Cross-environment splits by *recording date* (each of the 15 archives
  corresponds to a session in one of 3 rooms). The mapping date → room
  is not in the IEEE-DataPort metadata; the loader exposes `test_dates`
  as the cross-environment criterion and the user is expected to verify
  the room mapping per the docs/08 section 5 note.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

CSI_T = 100
CSI_S = 30
CSI_A = 3
NUM_CLASSES = 6


class StubCSI(Dataset):
    """Random CSI samples — pipeline plumbing only, no real signal."""

    def __init__(
        self,
        num_samples: int = 10,
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


# ---------------------------------------------------------------------------
# Widar3.0 cross-environment loader (T2.2)
#
# Filename schema: userN-G-P-O-T-rR.dat. The date (and thus the room) is
# captured by the parent directory, not the filename.

WIDAR3_FILENAME_RE = re.compile(
    r"^user(?P<user>\d+)"
    r"-(?P<gesture>\d+)"
    r"-(?P<position>\d+)"
    r"-(?P<orientation>\d+)"
    r"-(?P<instance>\d+)"
    r"-r(?P<receiver>\d+)\.dat$"
)


def parse_widar3_filename(name: str) -> dict:
    """Parse a Widar3.0 .dat filename into its integer slots."""
    m = WIDAR3_FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a Widar3.0 filename: {name!r}")
    return {k: int(v) for k, v in m.groupdict().items()}


def _walk_widar3_root(root: Path) -> list[tuple[Path, dict, str]]:
    """Walk `root/<date>/<user>/userN-...-rR.dat` recursively.

    Returns `(file_path, parsed_metadata, date_string)` for each match.
    `date_string` is the immediate parent of the user directory (e.g.
    "20181128"); used by `Widar3CrossEnvironment` as the split key.
    """
    items: list[tuple[Path, dict, str]] = []
    for p in root.rglob("*.dat"):
        try:
            meta = parse_widar3_filename(p.name)
        except ValueError:
            continue
        date = p.parent.parent.name
        items.append((p, meta, date))
    return items


def csi_complex_to_real(csi: torch.Tensor) -> torch.Tensor:
    """Project complex CSI onto a real-valued tensor via magnitude."""
    return csi.abs().to(torch.float32)


def _crop_or_pad_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Crop or zero-pad along the time axis (axis 0) to `target_t`."""
    t = x.shape[0]
    if t == target_t:
        return x
    if t > target_t:
        return x[:target_t]
    pad_shape = (target_t - t,) + tuple(x.shape[1:])
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


def load_widar3_dat(
    path: Path,
    time_steps: int = CSI_T,
    nrxnum: int = 3,
    ntxnum: int = 1,
) -> torch.Tensor:
    """Parse one Widar3.0 .dat into a `(time_steps, S, A)` real tensor."""
    import csiread

    parser = csiread.Intel(str(path), nrxnum=nrxnum, ntxnum=ntxnum)
    parser.read()
    csi = torch.from_numpy(parser.get_scaled_csi())
    if csi.ndim == 4:
        csi = csi.reshape(csi.shape[0], csi.shape[1], -1)
    real = csi_complex_to_real(csi)
    return _crop_or_pad_time(real, time_steps)


class Widar3CrossEnvironment(Dataset):
    """Widar3.0 cross-environment gesture-recognition dataset.

    Splits by recording date (parent-of-user directory). The 15 dates in
    the IEEE-DataPort release span 3 rooms (classroom, office, hall) per
    the Widar3.0 paper, but the date → room mapping isn't in the
    archive's metadata. Caller supplies `test_dates` (a list of date
    strings like `"20181128"`); train gets the complement.

    Args:
        root: directory containing the extracted Widar3.0 release
            (each immediate subdir is a date).
        train: True for the training split; False for the held-out test.
        test_dates: list of date strings held out as the test set.
            If None, defaults to a single date — set this to whatever
            corresponds to one full room in your extracted subset.
        time_steps: fixed length all samples are cropped/padded to.
        num_classes: number of gesture classes (1-indexed in Widar3.0
            files; mapped down to 0-indexed labels).
        cache_path: optional parsed-tensor cache.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        test_dates: list[str] | None = None,
        time_steps: int = CSI_T,
        num_classes: int = NUM_CLASSES,
        cache_path: str | Path | None = None,
        receivers: list[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.test_dates = list(test_dates) if test_dates else []
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.receivers = list(self.DEFAULT_RECEIVERS if receivers is None else receivers)
        self.cache_path = Path(cache_path) if cache_path else None

        cache_meta = {
            "train": self.train,
            "test_dates": sorted(self.test_dates),
            "time_steps": self.time_steps,
            "num_classes": self.num_classes,
            "receivers": sorted(self.receivers),
        }

        if self.cache_path is not None and self.cache_path.exists():
            blob = torch.load(self.cache_path, weights_only=False)
            cached_meta = blob.get("meta", {})
            if cached_meta != cache_meta:
                raise ValueError(
                    f"cache at {self.cache_path} was built with {cached_meta!r} "
                    f"but loader requested {cache_meta!r}"
                )
            self._x = blob["x"]
            self._y = blob["y"]
            return

        items = _walk_widar3_root(self.root)
        if self.train:
            items = [it for it in items if it[2] not in self.test_dates]
        else:
            items = [it for it in items if it[2] in self.test_dates]
        items = [it for it in items if 1 <= it[1]["gesture"] <= self.num_classes]
        items = [it for it in items if it[1]["receiver"] in self.receivers]

        if len(items) == 0:
            raise RuntimeError(
                f"no Widar3.0 .dat files found under {self.root!r} matching "
                f"the requested split (train={self.train}, "
                f"test_dates={self.test_dates}); check the data root path"
            )

        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for path, meta, _date in items:
            xs.append(load_widar3_dat(path, time_steps=self.time_steps))
            ys.append(meta["gesture"] - 1)

        self._x = torch.stack(xs, dim=0)
        self._y = torch.tensor(ys, dtype=torch.long)

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"x": self._x, "y": self._y, "meta": cache_meta},
                self.cache_path,
            )

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x[i], int(self._y[i].item())
