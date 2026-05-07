"""Smoke test: the T1.1 pipeline runs end-to-end on stub data without crashing."""

from __future__ import annotations

from src.slices.george.run import main


def test_smoke_runs_end_to_end():
    acc = main(seed=42, epochs=2, batch_size=4)
    assert 0.0 <= acc <= 1.0
