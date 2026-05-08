"""Smoke test: the T3.1 pipeline runs end-to-end on stub data without crashing."""

from __future__ import annotations

from src.slices.collins.run import main


def test_smoke_runs_end_to_end():
    result = main(seed=42, epochs=2, batch_size=4)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["tag"] == "T3.1-stub"
    assert len(result["pretrain_losses"]) == 2
