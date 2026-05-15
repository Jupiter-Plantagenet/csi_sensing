"""Production-run utilities for CSI experiments.

This module intentionally stays lightweight: it writes the required result
artifacts and can aggregate seed-level metrics already saved to disk. The
expensive training loops remain in the slice run modules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

CANONICAL_SEEDS = [42, 1337, 2024]
SANITY_GATE_CHANCE = 1.0 / 6.0
SANITY_GATE_MARGIN = 0.03
EXACT_REPRODUCTION_TOLERANCE = 0.001  # 0.1 percentage point in accuracy units.


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def write_result_bundle(
    result_dir: str | Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    notes: str,
    split_audit: dict[str, Any] | None = None,
) -> None:
    """Write the standard production result artifact set."""
    path = Path(result_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(_to_simple_yaml(config), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (path / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
    (path / "notes.md").write_text(notes.rstrip() + "\n", encoding="utf-8")
    if split_audit is not None:
        (path / "split_audit.json").write_text(
            json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8"
        )


def _to_simple_yaml(value: Any, indent: int = 0) -> str:
    """Small YAML writer for configs made of dict/list/scalar values."""
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key in sorted(value):
            child = value[key]
            if isinstance(child, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_to_simple_yaml(child, indent + 2).rstrip())
            else:
                lines.append(f"{pad}{key}: {_format_yaml_scalar(child)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{pad}- {_format_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{pad}{_format_yaml_scalar(value)}\n"


def _format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#[]{}&*!,|>'\"%@`"):
        return json.dumps(text)
    return text


def aggregate_metric(result_dirs: list[str | Path], metric_key: str) -> dict[str, Any]:
    """Aggregate one scalar metric across seed result directories."""
    values: list[float] = []
    sources: list[str] = []
    for result_dir in result_dirs:
        metrics_path = Path(result_dir) / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metric_key not in metrics:
            raise KeyError(f"{metrics_path} does not contain metric {metric_key!r}")
        values.append(float(metrics[metric_key]))
        sources.append(str(metrics_path))
    return {
        "metric": metric_key,
        "n": len(values),
        "values": values,
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "sources": sources,
    }


def sanity_gate_passed(acc: float) -> bool:
    return acc >= SANITY_GATE_CHANCE + SANITY_GATE_MARGIN


def classify_reproduction(
    reproduced: float,
    published: float,
    *,
    tolerance: float = EXACT_REPRODUCTION_TOLERANCE,
    hardware_impossible: bool = False,
) -> dict[str, Any]:
    """Classify a baseline reproduction against a published headline value.

    Accuracies are represented as fractions in [0, 1]. A tolerance of 0.001 is
    0.1 percentage point. Hardware-limited results are not "exact"; they are
    explicitly separated so the paper cannot accidentally present them as
    successful reproductions.
    """
    gap = abs(float(reproduced) - float(published))
    if gap <= tolerance:
        status = "exact"
    elif hardware_impossible:
        status = "hardware-limited"
    else:
        status = "failed"
    return {
        "reproduced": float(reproduced),
        "published": float(published),
        "gap": gap,
        "tolerance": tolerance,
        "within_tolerance": gap <= tolerance,
        "hardware_impossible": bool(hardware_impossible),
        "status": status,
    }


def default_result_dir(owner: str, method: str, *, date: str | None = None) -> str:
    date = date or datetime.now().strftime("%Y-%m-%d")
    return f"results/{date}-{owner}-{method}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate saved production metrics.")
    parser.add_argument("--metric", required=True)
    parser.add_argument("result_dirs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(aggregate_metric(args.result_dirs, args.metric), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
