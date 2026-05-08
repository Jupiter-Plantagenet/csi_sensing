"""T3.6 — Generic baseline vs phase-noise injection, single seed.

Runs ``run.main`` twice on cross-subject Widar3.0 with identical hyper-
parameters and seed; the only difference between arms is the SimCLR
``augment_fn``. Writes a results directory per docs/04 §7 and docs/07's
"results layout" convention:

    results/<date>-collins-T3.6-baseline-vs-phase-noise/
        config.yaml
        git_hash.txt
        metrics.json
        notes.md

The "candidate beats baseline" decision rule from docs/07 §2 (the comparison
rule) is properly multi-seed; T3.6 reports the single-seed numbers, T3.7
adds the seed sweep + paired test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from . import run as run_module

_DEFAULT_SLUG = "collins-T3.6-baseline-vs-phase-noise"


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"<git rev-parse failed: {exc}>"


def _serialize_config_yaml(config: dict) -> str:
    """Tiny YAML serializer for our flat config dicts.

    Avoiding the PyYAML dependency for a five-key dict; we control the keys
    so a sticky `repr`-based emitter is fine here.
    """
    lines = []
    for k, v in config.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv!r}")
        else:
            lines.append(f"{k}: {v!r}")
    return "\n".join(lines) + "\n"


def compare(
    data_root: str,
    cache_dir: str,
    phase_profile: str,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 64,
    max_files: int | None = 8000,
    results_root: str = "results",
    slug: str = _DEFAULT_SLUG,
    today: date | None = None,
) -> dict:
    """Run the two arms in sequence, write the results dir, return the metrics."""
    common = dict(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        data_root=data_root,
        cache_dir=cache_dir,
        max_files=max_files,
    )

    # Arm 1: generic baseline (Gaussian noise + subcarrier dropout).
    print("=" * 70)
    print("[T3.6] arm 1/2: generic baseline")
    print("=" * 70)
    baseline = run_module.main(**common, phase_profile=None)

    # Arm 2: phase-noise injection from the T3.3 profile.
    print("=" * 70)
    print("[T3.6] arm 2/2: phase-noise injection")
    print("=" * 70)
    phase_noise = run_module.main(**common, phase_profile=phase_profile)

    delta = float(phase_noise["accuracy"] - baseline["accuracy"])

    metrics = {
        "slice": "collins-T3.6",
        "comparison": "phase_noise vs generic_baseline",
        "seed": seed,
        "single_seed_caveat": (
            "single-seed result — paired multi-seed comparison comes in T3.7. "
            "Per docs/07 §2 the 'real improvement' rule is multi-seed only."
        ),
        "baseline": baseline,
        "phase_noise": phase_noise,
        "delta_accuracy_pp": round(100.0 * delta, 4),
    }

    # Results dir
    today = today or date.today()
    out_dir = Path(results_root) / f"{today.isoformat()}-{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "git_hash.txt").write_text(_git_hash() + "\n")
    (out_dir / "config.yaml").write_text(
        _serialize_config_yaml(
            {
                "common": common,
                "phase_profile": phase_profile,
                "results_dir": str(out_dir),
            }
        )
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Stub notes.md so a human can fill in the takeaway.
    notes = (
        f"# T3.6 — single-seed comparison\n\n"
        f"Date: {today.isoformat()}\n"
        f"Seed: {seed}\n\n"
        f"## Numbers\n\n"
        f"| arm | linear-probe acc | pretrain final loss |\n"
        f"|---|---|---|\n"
        f"| generic baseline | {baseline['accuracy']:.4f} | "
        f"{baseline['pretrain_losses'][-1]:.4f} |\n"
        f"| phase-noise inj  | {phase_noise['accuracy']:.4f} | "
        f"{phase_noise['pretrain_losses'][-1]:.4f} |\n"
        f"| delta (phase - baseline) | {100 * delta:+.2f} pp | — |\n\n"
        "## What was expected\n\n"
        "Filled in by the human after looking at the numbers.\n"
        "(Hypothesis from docs/03 §4.2: phase-noise injection helps cross-*chipset*\n"
        "transfer; cross-subject is a *robustness* stress-test of Stage A and was\n"
        "not designed to benefit from phase-noise. A near-zero delta is consistent\n"
        "with the hypothesis; a strongly negative delta would warrant attention.)\n\n"
        "## What we saw\n\n"
        "TODO: human takeaway.\n"
    )
    (out_dir / "notes.md").write_text(notes)

    # Final summary table to stdout.
    print()
    print("=" * 70)
    print("[T3.6] comparison summary")
    print("=" * 70)
    print(
        f"  generic baseline   : {baseline['accuracy']:.4f}    " f"({baseline['tag']})"
    )
    print(
        f"  phase-noise inject : {phase_noise['accuracy']:.4f}    "
        f"({phase_noise['tag']})"
    )
    print(f"  delta              : {100 * delta:+.2f} pp")
    print(f"  results dir        : {out_dir}")
    print("=" * 70)

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T3.6 single-seed paired comparison: baseline vs phase-noise"
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data/widar3/extracted",
        help="parent of YYYYMMDD/userN/*.dat",
    )
    p.add_argument("--cache-dir", type=str, default="data/widar3/cache")
    p.add_argument(
        "--phase-profile",
        type=str,
        default="data/widar3/cache/phase_profile.npz",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-files", type=int, default=8000)
    p.add_argument("--results-root", type=str, default="results")
    p.add_argument("--slug", type=str, default=_DEFAULT_SLUG)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        phase_profile=args.phase_profile,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_files=args.max_files,
        results_root=args.results_root,
        slug=args.slug,
    )
