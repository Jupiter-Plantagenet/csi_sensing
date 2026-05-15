"""Tests for src.slices.josiah.widar_bvp."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.slices.josiah.widar_bvp import (
    BVP_NORM_MEAN,
    BVP_NORM_STD,
    BVP_T,
    BVP_VX,
    BVP_VY,
    WidarBVP,
    audit_bvp_split,
    load_bvp_csv,
    parse_bvp_filename,
)


def test_parse_bvp_filename_canonical():
    meta = parse_bvp_filename("user5-3-1-2-7-1-1e-07-100-20-100000-L0.csv")
    assert meta == {
        "user": 5,
        "gesture": 3,
        "position": 1,
        "orientation": 2,
        "instance": 7,
        "receiver": 1,
    }


def test_parse_bvp_filename_rejects_garbage():
    with pytest.raises(ValueError):
        parse_bvp_filename("garbage.csv")


def _write_bvp_csv(path: Path, value: float = 0.0125) -> None:
    arr = np.full((BVP_T, BVP_VX * BVP_VY), value, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row.tolist())


def test_load_bvp_csv_shape_and_normalization(tmp_path: Path):
    csv_path = tmp_path / "user1-1-1-1-1-1-...csv"
    _write_bvp_csv(csv_path, value=BVP_NORM_MEAN + BVP_NORM_STD)
    out = load_bvp_csv(csv_path)
    assert out.shape == (BVP_T, BVP_VX, BVP_VY)
    assert out.dtype == torch.float32
    # (mean+std - mean)/std == 1.0
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def _make_mini_bvp_layout(tmp_path: Path) -> Path:
    """Build a tiny Widar BVP tree with two classes, four users, train+test."""
    root = tmp_path / "Widardata"
    # 2 folders -> 2 classes, with users 1..6 represented.
    for sub in ("train", "test"):
        for cls_name in ("1-Push&Pull", "2-Sweep"):
            for user in range(1, 7):
                fname = (
                    f"user{user}-{int(cls_name[0])}-1-1-1-1-"
                    "1e-07-100-20-100000-L0.csv"
                )
                _write_bvp_csv(root / sub / cls_name / fname, value=0.0)
    return root


def test_widar_bvp_cross_subject_split(tmp_path: Path):
    root = _make_mini_bvp_layout(tmp_path)
    train = WidarBVP(root, split="cross-subject", train=True, test_users=[1, 2])
    test = WidarBVP(root, split="cross-subject", train=False, test_users=[1, 2])
    # train uses users 3..6 over both train/ and test/ folders, 2 classes = 16
    # test uses users 1..2 over both folders, 2 classes = 8
    assert len(train) == 16
    assert len(test) == 8


def test_widar_bvp_sensefi_split(tmp_path: Path):
    root = _make_mini_bvp_layout(tmp_path)
    train = WidarBVP(root, split="sensefi", train=True)
    test = WidarBVP(root, split="sensefi", train=False)
    # sensefi split: all 12 train-folder samples and all 12 test-folder samples
    assert len(train) == 12
    assert len(test) == 12


def test_widar_bvp_gesture_filter(tmp_path: Path):
    root = _make_mini_bvp_layout(tmp_path)
    train_g1_only = WidarBVP(
        root, split="cross-subject", train=True, test_users=[1, 2], gesture_filter=[1]
    )
    # Only "1-Push&Pull" kept; users 3..6 over train+test folders -> 8 samples.
    assert len(train_g1_only) == 8


def test_audit_bvp_split_total(tmp_path: Path):
    root = _make_mini_bvp_layout(tmp_path)
    audit = audit_bvp_split(root, split="cross-subject", train=True, test_users=[1, 2])
    # 4 train users (3..6) x 2 classes x 2 subfolders (train+test) = 16; per user = 4.
    assert audit["total"] == 16
    assert audit["by_user"] == {3: 4, 4: 4, 5: 4, 6: 4}
