"""Widar3.0 BVP (Body-coordinate Velocity Profile) loader.

The BVP representation is the canonical published-baseline input for AutoFi
(Yang et al. 2022, paper §IV-D) and the SenseFi benchmark (Yang et al. 2023).
Each sample is a (time x vx x vy) = (22, 20, 20) real tensor extracted from
fused multi-receiver CSI via the Widar3.0 offline pipeline (Zhang et al. 2021).

Raw-CSI cross-subject on Widar3.0 sits at chance with any receiver subsample
in [1], [1,2,3], [1,2,3,4,5,6] (see results/2026-05-15-cross-subject-floor-finding.md
and the Gate 1 sweep on 2026-05-15). BVP is the published representation that
makes the baseline figures reproducible.

Filename schema: ``userN-G-T-P-O-INSTANCE-...csv`` (e.g. ``user1-5-1-1-13-1-...``).
Folders are class names: ``1-Push&Pull``, ``2-Sweep``, ... ``22-Draw-10``.

Two splits are supported:

* ``cross-subject``  — train users 5-17 / test users 1-4, matching the
  canonical project protocol in docs/09-execution-roadmap.md.
* ``sensefi``        — use the released ``Widardata/train/`` and
  ``Widardata/test/`` folders verbatim. This is the protocol used by the
  AutoFi authors' own SenseFi reference implementation; reproduces the
  benchmark cell that we can label "exact" or "hardware-limited" against.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

BVP_T = 22
BVP_VX = 20
BVP_VY = 20
BVP_NORM_MEAN = 0.0025
BVP_NORM_STD = 0.0119

NUM_BVP_CLASSES = 22

CANONICAL_CROSS_SUBJECT_TEST_USERS = [1, 2, 3, 4]

Split = Literal["cross-subject", "sensefi"]

# user-G-T-P-O-INSTANCE-... .csv
_BVP_FILENAME_RE = re.compile(
    r"^user(?P<user>\d+)"
    r"-(?P<gesture>\d+)"
    r"-(?P<position>\d+)"
    r"-(?P<orientation>\d+)"
    r"-(?P<instance>\d+)"
    r"-(?P<receiver>\d+)"
)


def parse_bvp_filename(name: str) -> dict[str, int]:
    """Parse ``userN-G-T-P-O-INSTANCE-rX-...csv`` into integer metadata."""
    m = _BVP_FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a Widar BVP filename: {name!r}")
    return {k: int(v) for k, v in m.groupdict().items()}


def _class_name_to_index(folder_name: str) -> int:
    """Folder ``"10-Draw-Zigzag(V)"`` -> class index ``9`` (zero-based)."""
    head = folder_name.split("-", 1)[0]
    return int(head) - 1


def load_bvp_csv(path: str | Path) -> torch.Tensor:
    """Load one BVP CSV and return a ``(22, 20, 20)`` float tensor.

    Normalization matches the SenseFi release (Yang et al. 2023,
    ``dataset.py::Widar_Dataset``): ``(x - 0.0025) / 0.0119``.
    """
    arr = np.genfromtxt(path, delimiter=",")
    arr = (arr - BVP_NORM_MEAN) / BVP_NORM_STD
    arr = arr.reshape(BVP_T, BVP_VX, BVP_VY)
    return torch.from_numpy(arr).float()


def _walk_split_dir(root: Path) -> list[tuple[Path, int, int]]:
    """Walk ``root/<class-name>/*.csv``; return ``[(path, class_idx, user)]``."""
    items: list[tuple[Path, int, int]] = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_idx = _class_name_to_index(cls_dir.name)
        for csv_path in cls_dir.glob("*.csv"):
            try:
                meta = parse_bvp_filename(csv_path.name)
            except ValueError:
                continue
            items.append((csv_path, cls_idx, meta["user"]))
    return items


class WidarBVP(Dataset):
    """Widar3.0 BVP dataset, cached to a single tensor file."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: Split = "cross-subject",
        train: bool = True,
        test_users: list[int] | None = None,
        gesture_filter: list[int] | None = None,
        cache_path: str | Path | None = None,
        max_files: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.train = train
        self.test_users = list(test_users or CANONICAL_CROSS_SUBJECT_TEST_USERS)
        self.gesture_filter = list(gesture_filter) if gesture_filter else None
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_files = max_files

        cache_meta = {
            "split": split,
            "train": train,
            "test_users": sorted(self.test_users),
            "gesture_filter": (
                sorted(self.gesture_filter) if self.gesture_filter else None
            ),
            "max_files": max_files,
            "format": "widar-bvp-22x20x20-norm-0.0025-0.0119",
        }
        self.meta = cache_meta

        if self.cache_path is not None and self.cache_path.exists():
            blob = torch.load(self.cache_path, weights_only=False)
            if blob.get("meta") != cache_meta:
                raise ValueError(
                    f"cache at {self.cache_path} built with {blob.get('meta')!r}, "
                    f"loader requested {cache_meta!r}"
                )
            self._x = blob["x"]
            self._y = blob["y"]
            self._users = blob["users"]
            return

        items = self._collect_items()
        if not items:
            raise RuntimeError(
                f"no BVP CSVs found under {self.root!r} matching {cache_meta!r}"
            )
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        users: list[int] = []
        for path, cls_idx, user in items:
            xs.append(load_bvp_csv(path))
            ys.append(cls_idx)
            users.append(user)
        self._x = torch.stack(xs, dim=0)
        self._y = torch.tensor(ys, dtype=torch.long)
        self._users = torch.tensor(users, dtype=torch.long)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"x": self._x, "y": self._y, "users": self._users, "meta": cache_meta},
                self.cache_path,
            )

    def _collect_items(self) -> list[tuple[Path, int, int]]:
        if self.split == "sensefi":
            sub = "train" if self.train else "test"
            items = _walk_split_dir(self.root / sub)
        elif self.split == "cross-subject":
            items: list[tuple[Path, int, int]] = []
            for sub in ("train", "test"):
                d = self.root / sub
                if d.is_dir():
                    items.extend(_walk_split_dir(d))
            if self.train:
                items = [it for it in items if it[2] not in self.test_users]
            else:
                items = [it for it in items if it[2] in self.test_users]
        else:
            raise ValueError(f"unknown split: {self.split!r}")
        if self.gesture_filter is not None:
            keep = set(g - 1 for g in self.gesture_filter)
            items = [it for it in items if it[1] in keep]
        if self.max_files is not None and len(items) > self.max_files:
            items = items[: self.max_files]
        return items

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x[i], int(self._y[i].item())


def audit_bvp_split(
    root: str | Path,
    *,
    split: Split = "cross-subject",
    train: bool | None = None,
    test_users: list[int] | None = None,
    gesture_filter: list[int] | None = None,
) -> dict:
    """Count BVP CSVs under the requested split. Mirrors ``audit_widar3_split``."""
    test_users = list(test_users or CANONICAL_CROSS_SUBJECT_TEST_USERS)
    root_p = Path(root)
    if split == "sensefi":
        all_items: list[tuple[Path, int, int]] = []
        if train is None:
            for sub in ("train", "test"):
                all_items.extend(_walk_split_dir(root_p / sub))
        else:
            sub = "train" if train else "test"
            all_items = _walk_split_dir(root_p / sub)
    else:
        all_items = []
        for sub in ("train", "test"):
            d = root_p / sub
            if d.is_dir():
                all_items.extend(_walk_split_dir(d))
        if train is True:
            all_items = [it for it in all_items if it[2] not in test_users]
        elif train is False:
            all_items = [it for it in all_items if it[2] in test_users]
    if gesture_filter:
        keep = set(g - 1 for g in gesture_filter)
        all_items = [it for it in all_items if it[1] in keep]

    from collections import Counter

    by_class = Counter(it[1] for it in all_items)
    by_user = Counter(it[2] for it in all_items)
    return {
        "root": str(root_p),
        "split": split,
        "train": train,
        "test_users": sorted(test_users),
        "gesture_filter": sorted(gesture_filter) if gesture_filter else None,
        "total": len(all_items),
        "by_class": dict(sorted(by_class.items())),
        "by_user": dict(sorted(by_user.items())),
    }


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x) for x in text.split(",")]


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Audit Widar BVP splits.")
    parser.add_argument("--root", default="data/widar3/Widardata")
    parser.add_argument(
        "--split", choices=["cross-subject", "sensefi"], default="cross-subject"
    )
    parser.add_argument("--train", choices=["true", "false", "all"], default="all")
    parser.add_argument("--test-users", default="1,2,3,4")
    parser.add_argument("--gestures", default=None, help="CSV gesture filter, e.g. 1,2,3,4,5,6")
    args = parser.parse_args()
    train = None if args.train == "all" else args.train == "true"
    audit = audit_bvp_split(
        args.root,
        split=args.split,
        train=train,
        test_users=_parse_csv_ints(args.test_users),
        gesture_filter=_parse_csv_ints(args.gestures),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
