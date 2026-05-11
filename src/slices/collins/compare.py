"""T3.6 / T3.7 — Generic baseline vs phase-noise injection on cross-subject.

Runs ``run.main`` for each ``(seed, arm)`` combination on cross-subject
Widar3.0 with identical hyperparameters across arms. The only difference
between arms is the SimCLR ``augment_fn``; the only difference across runs
of the same arm is the seed (dataset shuffle is keyed on the loader's
``shuffle_seed=0`` and stays fixed across SimCLR seeds, so the comparison
is properly paired — same files, same order, only torch RNG varies).

Writes a results directory per docs/04 §7 / docs/07:

    results/<date>-<slug>/
        config.yaml
        git_hash.txt
        metrics.json
        notes.md

Single seed is the T3.6 contract; ``--seeds 42 43 44`` is the T3.7 contract
(adds the docs/07 §2 paired comparison rule:
``phase_mean > baseline_mean + 1*baseline_std AND delta_mean > 0``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from datetime import date
from pathlib import Path

from . import run as run_module


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"<git rev-parse failed: {exc}>"


def _serialize_config_yaml(config: dict) -> str:
    """Tiny YAML emitter for our flat config dicts.

    Avoiding the PyYAML dependency for a handful of keys; we control the keys
    so a ``repr``-based emitter is fine here.
    """
    lines: list[str] = []
    for k, v in config.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv!r}")
        elif isinstance(v, list):
            lines.append(f"{k}: {v!r}")
        else:
            lines.append(f"{k}: {v!r}")
    return "\n".join(lines) + "\n"


def _run_single_seed(
    seed: int,
    *,
    data_root: str,
    cache_dir: str | None,
    phase_profile: str,
    epochs: int,
    batch_size: int,
    max_files: int | None,
) -> dict:
    """Run both arms with one seed; return a per-seed record."""
    common = dict(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        data_root=data_root,
        cache_dir=cache_dir,
        max_files=max_files,
    )

    print("=" * 70)
    print(f"[compare] seed={seed} arm 1/2: generic baseline")
    print("=" * 70)
    baseline = run_module.main(**common, phase_profile=None)

    print("=" * 70)
    print(f"[compare] seed={seed} arm 2/2: phase-noise injection")
    print("=" * 70)
    phase_noise = run_module.main(**common, phase_profile=phase_profile)

    delta_pp = round(100.0 * (phase_noise["accuracy"] - baseline["accuracy"]), 4)
    return {
        "seed": seed,
        "baseline": baseline,
        "phase_noise": phase_noise,
        "delta_accuracy_pp": delta_pp,
    }


def _aggregate(per_seed: list[dict]) -> dict:
    """Mean / std and the docs/07 §2 paired decision rule."""
    baseline_accs = [r["baseline"]["accuracy"] for r in per_seed]
    phase_accs = [r["phase_noise"]["accuracy"] for r in per_seed]
    deltas = [r["delta_accuracy_pp"] for r in per_seed]

    n = len(per_seed)
    baseline_mean = statistics.fmean(baseline_accs)
    phase_mean = statistics.fmean(phase_accs)
    delta_mean = statistics.fmean(deltas)

    if n >= 2:
        baseline_std = statistics.stdev(baseline_accs)
        phase_std = statistics.stdev(phase_accs)
        delta_std = statistics.stdev(deltas)
    else:
        # Single seed has no notion of variance; report 0 and flag the caveat.
        baseline_std = 0.0
        phase_std = 0.0
        delta_std = 0.0

    # docs/07 §2 rule: "candidate's mean must beat baseline's mean + 1 *
    # std-dev band of the baseline AND the sign is positive". The `n>=3`
    # gate is ours: with fewer seeds the std estimate is too noisy to act on.
    real_improvement = (
        n >= 3 and (phase_mean > baseline_mean + baseline_std) and (delta_mean > 0)
    )

    return {
        "n_seeds": n,
        "baseline": {
            "accuracies": baseline_accs,
            "mean": round(baseline_mean, 6),
            "std": round(baseline_std, 6),
        },
        "phase_noise": {
            "accuracies": phase_accs,
            "mean": round(phase_mean, 6),
            "std": round(phase_std, 6),
        },
        "delta_pp": {
            "per_seed": deltas,
            "mean": round(delta_mean, 4),
            "std": round(delta_std, 4),
        },
        "decision_rule": (
            "phase_mean > baseline_mean + 1*baseline_std AND delta_mean > 0 "
            "(requires >= 3 seeds)"
        ),
        "real_improvement_per_doc07": real_improvement,
    }


def compare(
    data_root: str,
    cache_dir: str | None,
    phase_profile: str,
    seeds: list[int],
    epochs: int = 20,
    batch_size: int = 64,
    max_files: int | None = 8000,
    results_root: str = "results",
    slug: str | None = None,
    today: date | None = None,
) -> dict:
    """Run both arms for each seed; emit per-seed + aggregate; write the dir."""
    if not seeds:
        raise ValueError("compare(): seeds must be a non-empty list")

    today = today or date.today()
    if slug is None:
        slug = (
            "collins-T3.6-baseline-vs-phase-noise"
            if len(seeds) == 1
            else "collins-T3.7-multiseed-baseline-vs-phase-noise"
        )

    per_seed: list[dict] = []
    for s in seeds:
        per_seed.append(
            _run_single_seed(
                s,
                data_root=data_root,
                cache_dir=cache_dir,
                phase_profile=phase_profile,
                epochs=epochs,
                batch_size=batch_size,
                max_files=max_files,
            )
        )

    agg = _aggregate(per_seed)

    metrics = {
        "slice": "collins-T3.7" if len(seeds) > 1 else "collins-T3.6",
        "comparison": "phase_noise vs generic_baseline",
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": agg,
    }

    out_dir = Path(results_root) / f"{today.isoformat()}-{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "git_hash.txt").write_text(_git_hash() + "\n")
    (out_dir / "config.yaml").write_text(
        _serialize_config_yaml(
            {
                "common": dict(
                    epochs=epochs,
                    batch_size=batch_size,
                    data_root=data_root,
                    cache_dir=cache_dir,
                    max_files=max_files,
                ),
                "seeds": list(seeds),
                "phase_profile": phase_profile,
                "results_dir": str(out_dir),
            }
        )
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Auto-fill notes.md with a paired comparison table; keep "What we saw"
    # as a TODO marker so the human can write the takeaway after running.
    n = len(seeds)
    table_rows = [
        f"| {r['seed']} | {r['baseline']['accuracy']:.4f} | "
        f"{r['phase_noise']['accuracy']:.4f} | {r['delta_accuracy_pp']:+.2f} pp |"
        for r in per_seed
    ]
    table_block = "\n".join(table_rows)
    notes = (
        f"# {'T3.7 — three-seed' if n > 1 else 'T3.6 — single-seed'} comparison\n\n"
        f"Date: {today.isoformat()}\n"
        f"Seeds: {list(seeds)}\n\n"
        "## Per-seed numbers\n\n"
        "| seed | generic baseline | phase-noise inj | delta (phase − baseline) |\n"
        "|---|---|---|---|\n"
        f"{table_block}\n\n"
        "## Aggregate (paired)\n\n"
        f"- generic baseline:  mean={agg['baseline']['mean']:.4f}  "
        f"std={agg['baseline']['std']:.4f}\n"
        f"- phase-noise inj:   mean={agg['phase_noise']['mean']:.4f}  "
        f"std={agg['phase_noise']['std']:.4f}\n"
        f"- delta:             mean={agg['delta_pp']['mean']:+.2f} pp  "
        f"std={agg['delta_pp']['std']:.2f} pp\n\n"
        "## docs/07 §2 decision rule\n\n"
        f"> phase_mean > baseline_mean + 1*baseline_std AND delta_mean > 0  "
        f"(requires ≥ 3 seeds)\n\n"
        f"**real improvement: "
        f"{'YES' if agg['real_improvement_per_doc07'] else 'NO'}**\n\n"
        "## What was expected\n\n"
        "Per docs/03 §4.2 phase-noise injection is the *cross-chipset* "
        "augmentation; cross-subject is a robustness stress-test of Stage A "
        "and is not the shift the augmentation was designed for. A near-zero "
        "or slightly negative delta is consistent with the hypothesis. The "
        "slice's headline causal claim still lives at Stage B (cross-chipset "
        "on CSI-Bench, follow-up issues).\n\n"
        "## What we saw\n\n"
        "TODO: human takeaway.\n"
    )
    (out_dir / "notes.md").write_text(notes)

    print()
    print("=" * 70)
    print(f"[compare] summary across {n} seed(s)")
    print("=" * 70)
    print(
        f"  generic baseline   : "
        f"{agg['baseline']['mean']:.4f} ± {agg['baseline']['std']:.4f}"
    )
    print(
        f"  phase-noise inject : "
        f"{agg['phase_noise']['mean']:.4f} ± {agg['phase_noise']['std']:.4f}"
    )
    print(
        f"  delta (phase − baseline) : "
        f"{agg['delta_pp']['mean']:+.2f} ± {agg['delta_pp']['std']:.2f} pp"
    )
    print(
        f"  real improvement (docs/07 §2): "
        f"{'YES' if agg['real_improvement_per_doc07'] else 'NO'}"
    )
    print(f"  results dir        : {out_dir}")
    print("=" * 70)

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired comparison driver: generic baseline vs phase-noise injection"
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
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="seeds to sweep; len 1 is the T3.6 single-seed mode, "
        "len >= 3 is the T3.7 multi-seed mode (default).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-files", type=int, default=8000)
    p.add_argument("--results-root", type=str, default="results")
    p.add_argument(
        "--slug",
        type=str,
        default=None,
        help="results dir slug; if omitted, auto-pick T3.6 / T3.7 based on len(seeds).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        phase_profile=args.phase_profile,
        seeds=list(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_files=args.max_files,
        results_root=args.results_root,
        slug=args.slug,
    )
