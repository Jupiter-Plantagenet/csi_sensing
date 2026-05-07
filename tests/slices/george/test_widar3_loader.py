"""Unit tests for the Widar3.0 cross-subject loader.

The data-dependent parts (csiread parsing, end-to-end loader smoke test on
real samples) are exercised in T1.6 once the dataset is in hand. These
tests cover the parts that work without real `.dat` files: filename
parsing, time-axis crop/pad, and complex→real projection.
"""

from __future__ import annotations

import pytest
import torch

from src.slices.george.data import (
    _crop_or_pad_time,
    csi_complex_to_real,
    parse_widar3_filename,
)


def test_parse_widar3_filename_canonical():
    meta = parse_widar3_filename("user1-5-4-2-15-r5.dat")
    assert meta == {
        "user": 1,
        "gesture": 5,
        "position": 4,
        "orientation": 2,
        "instance": 15,
        "receiver": 5,
    }


def test_parse_widar3_filename_multidigit_user():
    meta = parse_widar3_filename("user12-1-1-1-1-r1.dat")
    assert meta["user"] == 12


def test_parse_widar3_filename_rejects_garbage():
    with pytest.raises(ValueError):
        parse_widar3_filename("notawidar3file.dat")
    with pytest.raises(ValueError):
        parse_widar3_filename("user1-5-4-2-15.dat")  # missing -rR
    with pytest.raises(ValueError):
        parse_widar3_filename("config.cfg")


def test_crop_or_pad_time_crops():
    x = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1)
    out = _crop_or_pad_time(x, target_t=5)
    assert out.shape == (5, 1, 1)
    assert torch.equal(out.squeeze(), torch.arange(5, dtype=torch.float32))


def test_crop_or_pad_time_pads_with_zeros():
    x = torch.ones(3, 2, 4)
    out = _crop_or_pad_time(x, target_t=10)
    assert out.shape == (10, 2, 4)
    assert torch.all(out[:3] == 1)
    assert torch.all(out[3:] == 0)


def test_crop_or_pad_time_passthrough_on_match():
    x = torch.randn(7, 4, 2)
    out = _crop_or_pad_time(x, target_t=7)
    assert torch.equal(out, x)


def test_csi_complex_to_real_uses_magnitude():
    z = torch.complex(torch.tensor([3.0]), torch.tensor([4.0])).reshape(1, 1, 1)
    out = csi_complex_to_real(z)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([[[5.0]]]))
