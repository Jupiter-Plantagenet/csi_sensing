"""Shared Widar3.0 raw-CSI loading and auditing helpers.

The slice directories keep their public ``data.py`` entry points, but this
module centralises the production-critical details: canonical filters,
real/imag representation, cache metadata, and split auditing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch.utils.data import Dataset

CSI_T = 200
CSI_S = 30
CSI_A = 3
NUM_CLASSES = 6

CANONICAL_TEST_SUBJECTS = [1, 2, 3, 4]
CANONICAL_GESTURES = [1, 2, 3, 4, 5, 6]
CANONICAL_POSITIONS = [1]
CANONICAL_ORIENTATIONS = [1]
CANONICAL_RECEIVERS = [1]
CANONICAL_REPRESENTATION = "real-imag"

Representation = Literal["real-imag", "magnitude"]

WIDAR3_FILENAME_RE = re.compile(
    r"^user(?P<user>\d+)"
    r"-(?P<gesture>\d+)"
    r"-(?P<position>\d+)"
    r"-(?P<orientation>\d+)"
    r"-(?P<instance>\d+)"
    r"-r(?P<receiver>\d+)\.dat$"
)


def parse_widar3_filename(name: str) -> dict[str, int]:
    """Parse a Widar3.0 filename into integer metadata fields."""
    m = WIDAR3_FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a Widar3.0 filename: {name!r}")
    return {k: int(v) for k, v in m.groupdict().items()}


def walk_widar3_root(root: str | Path) -> list[tuple[Path, dict[str, int], str]]:
    """Return ``(path, metadata, date)`` for every matching ``.dat`` file."""
    out: list[tuple[Path, dict[str, int], str]] = []
    for p in Path(root).rglob("*.dat"):
        try:
            meta = parse_widar3_filename(p.name)
        except ValueError:
            continue
        # Expected layout: root/<date>/userN/file.dat.
        date = p.parent.parent.name
        out.append((p, meta, date))
    return out


def csi_complex_to_real(
    csi: torch.Tensor,
    representation: Representation = CANONICAL_REPRESENTATION,
) -> torch.Tensor:
    """Project complex CSI to a real tensor.

    ``magnitude`` returns ``(T, S, A)``. ``real-imag`` concatenates real and
    imaginary channels along the antenna axis, returning ``(T, S, 2*A)``.
    """
    if representation == "magnitude":
        return csi.abs().to(torch.float32)
    if representation == "real-imag":
        return torch.cat([csi.real, csi.imag], dim=-1).to(torch.float32)
    raise ValueError(f"unknown representation: {representation!r}")


def crop_or_pad_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Crop or zero-pad along the time axis to ``target_t``."""
    t = x.shape[0]
    if t == target_t:
        return x
    if t > target_t:
        return x[:target_t]
    pad_shape = (target_t - t,) + tuple(x.shape[1:])
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


def load_widar3_dat(
    path: str | Path,
    *,
    time_steps: int = CSI_T,
    representation: Representation = CANONICAL_REPRESENTATION,
    nrxnum: int = 3,
    ntxnum: int = 1,
) -> torch.Tensor:
    """Parse one Widar3.0 ``.dat`` into a fixed-length real tensor."""
    import csiread

    parser = csiread.Intel(str(path), nrxnum=nrxnum, ntxnum=ntxnum)
    parser.read()
    csi = torch.from_numpy(parser.get_scaled_csi())
    if csi.numel() == 0 or csi.shape[0] == 0:
        raise ValueError(f"empty CSI in {path}")
    if csi.ndim == 4:
        csi = csi.reshape(csi.shape[0], csi.shape[1], -1)
    real = csi_complex_to_real(csi, representation=representation)
    return crop_or_pad_time(real, time_steps)


def _normalise_list(values: Iterable[int] | None, default: list[int]) -> list[int]:
    return list(default if values is None else values)


def _filter_items(
    items: list[tuple[Path, dict[str, int], str]],
    *,
    gestures: list[int],
    positions: list[int],
    orientations: list[int],
    receivers: list[int],
) -> list[tuple[Path, dict[str, int], str]]:
    return [
        item
        for item in items
        if item[1]["gesture"] in gestures
        and item[1]["position"] in positions
        and item[1]["orientation"] in orientations
        and item[1]["receiver"] in receivers
    ]


def _limit_items_balanced(
    items: list[tuple[Path, dict[str, int], str]],
    max_files: int | None,
) -> list[tuple[Path, dict[str, int], str]]:
    """Limit debug datasets without collapsing to a single gesture class."""
    if max_files is None or len(items) <= max_files:
        return items
    by_gesture: dict[int, list[tuple[Path, dict[str, int], str]]] = {}
    for item in items:
        by_gesture.setdefault(item[1]["gesture"], []).append(item)
    selected: list[tuple[Path, dict[str, int], str]] = []
    gestures = sorted(by_gesture)
    index = 0
    while len(selected) < max_files:
        progressed = False
        for gesture in gestures:
            bucket = by_gesture[gesture]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) >= max_files:
                    break
        if not progressed:
            break
        index += 1
    return selected


class Widar3CrossSubject(Dataset):
    """Canonical raw-CSI cross-subject loader."""

    DEFAULT_TEST_SUBJECTS = CANONICAL_TEST_SUBJECTS
    DEFAULT_GESTURES = CANONICAL_GESTURES
    DEFAULT_POSITIONS = CANONICAL_POSITIONS
    DEFAULT_ORIENTATIONS = CANONICAL_ORIENTATIONS
    DEFAULT_RECEIVERS = CANONICAL_RECEIVERS

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        test_subjects: list[int] | None = None,
        time_steps: int = CSI_T,
        num_classes: int = NUM_CLASSES,
        cache_path: str | Path | None = None,
        receivers: list[int] | None = None,
        positions: list[int] | None = None,
        orientations: list[int] | None = None,
        gestures: list[int] | None = None,
        representation: Representation = CANONICAL_REPRESENTATION,
        max_files: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.test_subjects = _normalise_list(test_subjects, self.DEFAULT_TEST_SUBJECTS)
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.gestures = _normalise_list(gestures, list(range(1, num_classes + 1)))
        self.positions = _normalise_list(positions, self.DEFAULT_POSITIONS)
        self.orientations = _normalise_list(orientations, self.DEFAULT_ORIENTATIONS)
        self.receivers = _normalise_list(receivers, self.DEFAULT_RECEIVERS)
        self.representation = representation
        self.max_files = max_files
        self.cache_path = Path(cache_path) if cache_path else None

        cache_meta = {
            "split": "cross-subject",
            "train": self.train,
            "test_subjects": sorted(self.test_subjects),
            "time_steps": self.time_steps,
            "num_classes": self.num_classes,
            "gestures": sorted(self.gestures),
            "positions": sorted(self.positions),
            "orientations": sorted(self.orientations),
            "receivers": sorted(self.receivers),
            "representation": self.representation,
            "max_files": self.max_files,
            "max_file_selection": "balanced-by-gesture-v1",
        }
        self.meta = cache_meta

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

        items = walk_widar3_root(self.root)
        if self.train:
            items = [it for it in items if it[1]["user"] not in self.test_subjects]
        else:
            items = [it for it in items if it[1]["user"] in self.test_subjects]
        items = _filter_items(
            items,
            gestures=self.gestures,
            positions=self.positions,
            orientations=self.orientations,
            receivers=self.receivers,
        )
        items = _limit_items_balanced(items, self.max_files)
        self._load_items(items, cache_meta)

    def _load_items(
        self,
        items: list[tuple[Path, dict[str, int], str]],
        cache_meta: dict,
    ) -> None:
        if not items:
            raise RuntimeError(
                f"no Widar3.0 .dat files found under {self.root!r} matching "
                f"{cache_meta!r}"
            )
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        skipped = 0
        for path, meta, _date in items:
            try:
                xs.append(
                    load_widar3_dat(
                        path,
                        time_steps=self.time_steps,
                        representation=self.representation,
                    )
                )
                ys.append(meta["gesture"] - 1)
            except ValueError:
                skipped += 1
        self.skipped_empty_dats = skipped
        if skipped:
            print(f"[Widar3] skipped {skipped} empty .dat files")
        if not xs:
            raise RuntimeError(f"every .dat under {self.root!r} was empty")
        self._x = torch.stack(xs, dim=0)
        self._y = torch.tensor(ys, dtype=torch.long)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"x": self._x, "y": self._y, "meta": cache_meta}, self.cache_path)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x[i], int(self._y[i].item())


class Widar3CrossEnvironment(Widar3CrossSubject):
    """Canonical raw-CSI cross-environment loader, split by recording date."""

    DEFAULT_TEST_DATES = ["20181128"]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        test_dates: list[str] | None = None,
        time_steps: int = CSI_T,
        num_classes: int = NUM_CLASSES,
        cache_path: str | Path | None = None,
        receivers: list[int] | None = None,
        positions: list[int] | None = None,
        orientations: list[int] | None = None,
        gestures: list[int] | None = None,
        representation: Representation = CANONICAL_REPRESENTATION,
        max_files: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.test_dates = list(self.DEFAULT_TEST_DATES if test_dates is None else test_dates)
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.gestures = _normalise_list(gestures, list(range(1, num_classes + 1)))
        self.positions = _normalise_list(positions, self.DEFAULT_POSITIONS)
        self.orientations = _normalise_list(orientations, self.DEFAULT_ORIENTATIONS)
        self.receivers = _normalise_list(receivers, self.DEFAULT_RECEIVERS)
        self.representation = representation
        self.max_files = max_files
        self.cache_path = Path(cache_path) if cache_path else None

        cache_meta = {
            "split": "cross-environment",
            "train": self.train,
            "test_dates": sorted(self.test_dates),
            "time_steps": self.time_steps,
            "num_classes": self.num_classes,
            "gestures": sorted(self.gestures),
            "positions": sorted(self.positions),
            "orientations": sorted(self.orientations),
            "receivers": sorted(self.receivers),
            "representation": self.representation,
            "max_files": self.max_files,
            "max_file_selection": "balanced-by-gesture-v1",
        }
        self.meta = cache_meta

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

        items = walk_widar3_root(self.root)
        if self.train:
            items = [it for it in items if it[2] not in self.test_dates]
        else:
            items = [it for it in items if it[2] in self.test_dates]
        items = _filter_items(
            items,
            gestures=self.gestures,
            positions=self.positions,
            orientations=self.orientations,
            receivers=self.receivers,
        )
        items = _limit_items_balanced(items, self.max_files)
        self._load_items(items, cache_meta)


def audit_widar3_split(
    root: str | Path,
    *,
    split: Literal["cross-subject", "cross-environment"] = "cross-subject",
    train: bool | None = None,
    test_subjects: list[int] | None = None,
    test_dates: list[str] | None = None,
    gestures: list[int] | None = None,
    positions: list[int] | None = None,
    orientations: list[int] | None = None,
    receivers: list[int] | None = None,
) -> dict:
    """Return counts for a requested Widar3.0 split/filter combination."""
    test_subjects = _normalise_list(test_subjects, CANONICAL_TEST_SUBJECTS)
    test_dates = list(Widar3CrossEnvironment.DEFAULT_TEST_DATES if test_dates is None else test_dates)
    gestures = _normalise_list(gestures, CANONICAL_GESTURES)
    positions = _normalise_list(positions, CANONICAL_POSITIONS)
    orientations = _normalise_list(orientations, CANONICAL_ORIENTATIONS)
    receivers = _normalise_list(receivers, CANONICAL_RECEIVERS)

    items = walk_widar3_root(root)
    if split == "cross-subject" and train is not None:
        if train:
            items = [it for it in items if it[1]["user"] not in test_subjects]
        else:
            items = [it for it in items if it[1]["user"] in test_subjects]
    elif split == "cross-environment" and train is not None:
        if train:
            items = [it for it in items if it[2] not in test_dates]
        else:
            items = [it for it in items if it[2] in test_dates]
    items = _filter_items(
        items,
        gestures=gestures,
        positions=positions,
        orientations=orientations,
        receivers=receivers,
    )

    counters = {
        "user": Counter(),
        "date": Counter(),
        "gesture": Counter(),
        "position": Counter(),
        "orientation": Counter(),
        "receiver": Counter(),
    }
    for _path, meta, date in items:
        counters["date"][date] += 1
        for key in ("user", "gesture", "position", "orientation", "receiver"):
            counters[key][meta[key]] += 1
    return {
        "root": str(root),
        "split": split,
        "train": train,
        "filters": {
            "test_subjects": sorted(test_subjects),
            "test_dates": sorted(test_dates),
            "gestures": sorted(gestures),
            "positions": sorted(positions),
            "orientations": sorted(orientations),
            "receivers": sorted(receivers),
        },
        "total": len(items),
        "counts": {
            key: dict(sorted(counter.items(), key=lambda kv: str(kv[0])))
            for key, counter in counters.items()
        },
    }


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None or text == "":
        return None
    return [int(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Widar3.0 raw-CSI splits.")
    parser.add_argument("--root", default="data/widar3/raw")
    parser.add_argument("--split", choices=["cross-subject", "cross-environment"], default="cross-subject")
    parser.add_argument("--train", choices=["true", "false", "all"], default="all")
    parser.add_argument("--test-subjects", default="1,2,3,4")
    parser.add_argument("--test-date", action="append", dest="test_dates", default=None)
    parser.add_argument("--gestures", default="1,2,3,4,5,6")
    parser.add_argument("--positions", default="1")
    parser.add_argument("--orientations", default="1")
    parser.add_argument("--receivers", default="1")
    args = parser.parse_args()

    train = None if args.train == "all" else args.train == "true"
    audit = audit_widar3_split(
        args.root,
        split=args.split,
        train=train,
        test_subjects=_parse_csv_ints(args.test_subjects),
        test_dates=args.test_dates,
        gestures=_parse_csv_ints(args.gestures),
        positions=_parse_csv_ints(args.positions),
        orientations=_parse_csv_ints(args.orientations),
        receivers=_parse_csv_ints(args.receivers),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
