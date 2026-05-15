"""Dataset wrappers for Slice 5 (baseline reproduction).

T5.1 ships `StubCSI` for end-to-end pipeline plumbing on random tensors.
T5.2 adds `Widar3CrossSubject` — the real Widar3.0 cross-subject loader,
ported from Slice 1's loader per the slice-independence convention in
`docs/08-team-work-plan.md` section 2.

Convention: each sample is a real-valued CSI tensor shaped `(T, S, A)` where
    T = time samples (per-instance variable; cropped/padded to a fixed length)
    S = subcarriers (30 for Intel 5300)
    A = antenna pairs (3 for the typical Widar3.0 1-TX × 3-RX setup)
plus a single integer gesture label in `[0, NUM_CLASSES)`.

The Widar3.0 raw release ships `.dat` files in the Linux 802.11n CSI Tool
format (Halperin et al.). We use `csiread.Intel` for parsing; complex CSI
is reduced to its magnitude as the real-valued projection consumed by the
encoder. T5.2 keeps the same projection as Slice 1 so the comparison row
in the team's results figure is apples-to-apples (same preprocessing,
different training objective).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

CSI_T = 200
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
# Widar3.0 cross-subject loader (T5.2)

WIDAR3_FILENAME_RE = re.compile(
    r"^user(?P<user>\d+)"
    r"-(?P<gesture>\d+)"
    r"-(?P<position>\d+)"
    r"-(?P<orientation>\d+)"
    r"-(?P<instance>\d+)"
    r"-r(?P<receiver>\d+)\.dat$"
)


def parse_widar3_filename(name: str) -> dict:
    """Parse a Widar3.0 .dat filename into its integer slots.

    Raises ValueError if the name doesn't match the expected pattern.
    """
    m = WIDAR3_FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a Widar3.0 filename: {name!r}")
    return {k: int(v) for k, v in m.groupdict().items()}


def _walk_widar3_root(root: Path) -> list[tuple[Path, dict]]:
    """Walk `root/<date>/<user>/userN-...-rR.dat` recursively.

    Returns a list of (file_path, parsed_metadata) for every .dat file whose
    name matches the Widar3.0 pattern. Files that don't match (e.g. .cfg
    config files, unrelated .dat) are silently skipped.
    """
    items: list[tuple[Path, dict]] = []
    for p in root.rglob("*.dat"):
        try:
            meta = parse_widar3_filename(p.name)
        except ValueError:
            continue
        items.append((p, meta))
    return items


def csi_complex_to_real(csi: torch.Tensor) -> torch.Tensor:
    """Project complex CSI onto a real-valued tensor via magnitude.

    Input shape: `(T, S, A)` complex; output shape: `(T, S, A)` real float32.
    Same projection as Slice 1 keeps the comparison row honest.
    """
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
    """Parse one Widar3.0 .dat into a `(time_steps, S, A)` real tensor.

    Uses `csiread.Intel` with the Widar3.0-standard 1-TX × 3-RX shape;
    csiread's library default `ntxnum=2` would silently produce 6 antenna
    channels and break the encoder.
    """
    import csiread

    parser = csiread.Intel(str(path), nrxnum=nrxnum, ntxnum=ntxnum)
    parser.read()
    csi = torch.from_numpy(parser.get_scaled_csi())  # (T, S, Nrx, Ntx) complex
    if csi.numel() == 0 or csi.shape[0] == 0:
        raise ValueError(f"empty CSI in {path}")
    if csi.ndim == 4:
        csi = csi.reshape(csi.shape[0], csi.shape[1], -1)
    real = csi_complex_to_real(csi)
    return _crop_or_pad_time(real, time_steps)


class Widar3CrossSubject(Dataset):
    """Widar3.0 cross-subject gesture-recognition dataset.

    Args mirror Slice 1's `Widar3CrossSubject`. The cross-subject split
    convention is shared (default test on user IDs 1-4) so Slice 5's
    supervised baseline reports a number directly comparable to Slice 1's
    Doppler-aware augmentation result. If the team later decides to change
    the canonical split, both slices update together.

    Args:
        root: directory containing Widar3.0 `.dat` files (recursively).
        train: True for the training split (subjects ∉ test_subjects);
            False for the held-out test split.
        test_subjects: list of user IDs held out as the cross-subject test
            set. Default `[1, 2, 3, 4]` matches Slice 1's convention.
        time_steps: fixed length all samples are cropped/padded to.
        num_classes: number of gesture classes.
        cache_path: optional path for a parsed-tensor cache. If set and the
            cache exists, the dataset loads from it instead of re-parsing
            every `.dat` on `__init__`.
    """

    DEFAULT_TEST_SUBJECTS: list[int] = [1, 2, 3, 4]
    DEFAULT_RECEIVERS: list[int] = [1]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        test_subjects: list[int] | None = None,
        time_steps: int = CSI_T,
        num_classes: int = NUM_CLASSES,
        cache_path: str | Path | None = None,
        receivers: list[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.test_subjects = list(
            self.DEFAULT_TEST_SUBJECTS if test_subjects is None else test_subjects
        )
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.receivers = list(self.DEFAULT_RECEIVERS if receivers is None else receivers)
        self.cache_path = Path(cache_path) if cache_path else None

        cache_meta = {
            "train": self.train,
            "test_subjects": sorted(self.test_subjects),
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
                    f"but loader requested {cache_meta!r}; pass a different "
                    "cache_path or delete the existing one"
                )
            self._x = blob["x"]
            self._y = blob["y"]
            return

        items = _walk_widar3_root(self.root)
        if self.train:
            items = [it for it in items if it[1]["user"] not in self.test_subjects]
        else:
            items = [it for it in items if it[1]["user"] in self.test_subjects]
        items = [it for it in items if 1 <= it[1]["gesture"] <= self.num_classes]
        items = [it for it in items if it[1]["receiver"] in self.receivers]

        if len(items) == 0:
            raise RuntimeError(
                f"no Widar3.0 .dat files found under {self.root!r}; check the "
                "data root path or extract a CSI archive first"
            )

        xs: list[torch.Tensor] = []
        ys: list[int] = []
        skipped = 0
        for path, meta in items:
            try:
                xs.append(load_widar3_dat(path, time_steps=self.time_steps))
                ys.append(meta["gesture"] - 1)  # to 0-indexed
            except ValueError:
                skipped += 1
        if skipped:
            print(f"[Widar3CrossSubject] skipped {skipped} empty .dat files")
        if not xs:
            raise RuntimeError(
                f"every .dat under {self.root!r} was empty; check the data"
            )

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
