"""Tests for production result helpers."""

from __future__ import annotations

import json

from src.production import (
    EXACT_REPRODUCTION_TOLERANCE,
    aggregate_metric,
    classify_reproduction,
    sanity_gate_passed,
    write_result_bundle,
)


def test_write_result_bundle_creates_required_files(tmp_path) -> None:
    out = tmp_path / "result"
    write_result_bundle(
        out,
        config={"seed": 42, "method": "supervised"},
        metrics={"accuracy": 0.25},
        notes="# Notes\n\nDiagnostic.",
        split_audit={"total": 3},
    )
    for name in ("config.yaml", "metrics.json", "git_hash.txt", "notes.md", "split_audit.json"):
        assert (out / name).exists()
    assert json.loads((out / "metrics.json").read_text())["accuracy"] == 0.25


def test_aggregate_metric_reads_seed_dirs(tmp_path) -> None:
    dirs = []
    for idx, acc in enumerate([0.2, 0.3, 0.4]):
        d = tmp_path / f"seed{idx}"
        write_result_bundle(
            d,
            config={"seed": idx},
            metrics={"accuracy": acc},
            notes="ok",
        )
        dirs.append(d)
    agg = aggregate_metric(dirs, "accuracy")
    assert agg["n"] == 3
    assert agg["mean"] == 0.3


def test_sanity_gate_margin() -> None:
    assert sanity_gate_passed(0.20)
    assert not sanity_gate_passed(0.18)


def test_classify_reproduction_exact_within_point_one_percentage_point() -> None:
    out = classify_reproduction(0.9009, 0.9000)
    assert out["status"] == "exact"
    assert out["within_tolerance"] is True
    assert out["tolerance"] == EXACT_REPRODUCTION_TOLERANCE


def test_classify_reproduction_failed_outside_tolerance() -> None:
    out = classify_reproduction(0.902, 0.900)
    assert out["status"] == "failed"
    assert out["within_tolerance"] is False


def test_classify_reproduction_hardware_limited_is_not_exact() -> None:
    out = classify_reproduction(0.86, 0.90, hardware_impossible=True)
    assert out["status"] == "hardware-limited"
    assert out["within_tolerance"] is False
