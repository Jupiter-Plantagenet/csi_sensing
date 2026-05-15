"""Unit tests for Slice 5's Widar3.0 cross-subject loader.

Data-dependent parts (csiread parsing, end-to-end smoke on real `.dat`
files) are exercised in T5.2's real-data run. These tests cover the
parts that don't need a Widar3.0 install: filename parsing, time-axis
crop/pad, and the complex->real projection.
"""

from __future__ import annotations

import pytest
import torch

from src.slices.josiah.data import (
    _crop_or_pad_time,
    csi_complex_to_real,
    parse_widar3_filename,
)


def test_parse_widar3_filename_canonical() -> None:
    meta = parse_widar3_filename("user1-5-4-2-15-r5.dat")
    assert meta == {
        "user": 1,
        "gesture": 5,
        "position": 4,
        "orientation": 2,
        "instance": 15,
        "receiver": 5,
    }


def test_parse_widar3_filename_multidigit_user() -> None:
    meta = parse_widar3_filename("user12-1-1-1-1-r1.dat")
    assert meta["user"] == 12


def test_parse_widar3_filename_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_widar3_filename("notawidar3file.dat")
    with pytest.raises(ValueError):
        parse_widar3_filename("user1-5-4-2-15.dat")  # missing -rR
    with pytest.raises(ValueError):
        parse_widar3_filename("config.cfg")


def test_crop_or_pad_time_crops() -> None:
    x = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1)
    out = _crop_or_pad_time(x, target_t=5)
    assert out.shape == (5, 1, 1)
    # The first 5 elements should be preserved.
    assert torch.equal(out, x[:5])


def test_crop_or_pad_time_pads() -> None:
    x = torch.ones((3, 2, 4), dtype=torch.float32)
    out = _crop_or_pad_time(x, target_t=7)
    assert out.shape == (7, 2, 4)
    # First 3 rows should be unchanged ones; remaining 4 are zero.
    assert torch.equal(out[:3], x)
    assert torch.equal(out[3:], torch.zeros((4, 2, 4)))


def test_crop_or_pad_time_identity() -> None:
    x = torch.randn(10, 3, 2)
    out = _crop_or_pad_time(x, target_t=10)
    assert torch.equal(out, x)


def test_csi_complex_to_real_projection() -> None:
    # Pure imaginary input has magnitude equal to the absolute value.
    z = torch.complex(torch.zeros(2, 2, 2), torch.tensor([[[3.0, 4.0]] * 2] * 2))
    out = csi_complex_to_real(z)
    assert out.dtype == torch.float32
    assert out.shape == z.shape
    expected = z.abs().to(torch.float32)
    assert torch.allclose(out, expected)
