"""Smoke test for Slice 5's T5.1 scaffold.

Confirms the supervised pipeline runs end-to-end on stub data without
crashing. The accuracy is not checked — stub labels are random, so a
correctness assertion would be against chance.
"""

from __future__ import annotations

from src.slices.josiah.run import main


def test_supervised_pipeline_runs() -> None:
    acc = main(seed=42, epochs=2, batch_size=4)
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"
