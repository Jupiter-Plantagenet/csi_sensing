"""Production runner for canonical raw-CSI experiments.

This is a thin orchestrator around the existing slice entry points. It does
not run automatically; call it explicitly when the sanity gate says production
runs are worth spending compute on.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.production import aggregate_metric, write_result_bundle
from src.slices.widar import audit_widar3_split


def _josiah(mode: str) -> Callable[..., float]:
    from src.slices.josiah.run import main

    return lambda **kwargs: main(mode=mode, **kwargs)


def _josiah_bvp(mode: str) -> Callable[..., float]:
    from src.slices.josiah.bvp_methods import run_bvp_project_method

    def runner(*, seed: int, epochs: int, batch_size: int, cache_dir: str, bvp_root: str, **_: dict) -> float:
        return run_bvp_project_method(
            mode,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            cache_dir=cache_dir,
            bvp_root=bvp_root,
        )

    return runner


def _josiah_autofi() -> Callable[..., float]:
    from src.slices.josiah.autofi import run_autofi

    def runner(*, seed: int, epochs: int, batch_size: int, cache_dir: str, bvp_root: str, **_: dict) -> float:
        # epochs here = SSL epochs (paper / SenseFi default 100); probe_epochs fixed to 300.
        return run_autofi(
            seed=seed,
            ssl_epochs=epochs,
            probe_epochs=300,
            batch_size=batch_size,
            cache_dir=cache_dir,
            bvp_root=bvp_root,
            protocol="sensefi",
        )

    return runner


def _josiah_mae() -> Callable[..., float]:
    from src.slices.josiah.mae import run_mae

    def runner(*, seed: int, epochs: int, batch_size: int, cache_dir: str, bvp_root: str, **_: dict) -> float:
        return run_mae(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            cache_dir=cache_dir,
            bvp_root=bvp_root,
        )

    return runner


def _josiah_autofi_uthar() -> Callable[..., float]:
    from src.slices.josiah.autofi import run_autofi_uthar

    def runner(*, seed: int, epochs: int, batch_size: int, cache_dir: str, ut_har_root: str = "data/ut_har/UT_HAR", **_: dict) -> float:
        return run_autofi_uthar(
            seed=seed,
            ssl_epochs=epochs,
            probe_epochs=300,
            batch_size=batch_size,
            cache_dir=cache_dir,
            ut_har_root=ut_har_root,
        )

    return runner


def _josiah_mae_uthar() -> Callable[..., float]:
    from src.slices.josiah.mae import run_mae_uthar

    def runner(*, seed: int, epochs: int, batch_size: int, cache_dir: str, ut_har_root: str = "data/ut_har/UT_HAR", **_: dict) -> float:
        return run_mae_uthar(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            cache_dir=cache_dir,
            ut_har_root=ut_har_root,
        )

    return runner


def _josiah_capc(pretrain_env: str = "home", eval_env: str = "home") -> Callable[..., float]:
    from src.slices.josiah.capc import run_capc

    def runner(
        *,
        seed: int,
        epochs: int,
        batch_size: int,
        cache_dir: str,
        signfi_root: str = "data",
        k_shot: int = 10,
        **_: dict,
    ) -> float:
        return run_capc(
            seed=seed,
            ssl_epochs=epochs,
            batch_size=batch_size,
            cache_dir=cache_dir,
            data_root=signfi_root,
            k_shot=k_shot,
            pretrain_env=pretrain_env,
            eval_env=eval_env,
        )

    return runner


def _george(aug: str) -> Callable[..., float]:
    from src.slices.george.run import main

    return lambda **kwargs: main(aug=aug, **kwargs)


def _chigozie(mode: str) -> Callable[..., float]:
    from src.slices.chigozie.run import main

    return lambda **kwargs: main(mode=mode, **kwargs)


def _ihunanya(mode: str) -> Callable[..., float]:
    from src.slices.ihunanya.run import main

    return lambda **kwargs: main(mode=mode, **kwargs)


def _victor(mode: str) -> Callable[..., float]:
    from src.slices.victor.run import main

    return lambda **kwargs: main(mode=mode, **kwargs)


METHODS: dict[str, tuple[str, Callable[..., float], str, str]] = {
    "supervised": ("josiah", _josiah("supervised"), "cross-subject", "project-baseline"),
    "simclr-trivial": ("josiah", _josiah("simclr-trivial"), "cross-subject", "project-baseline"),
    "simclr-handcrafted": ("josiah", _josiah("simclr-handcrafted"), "cross-subject", "project-baseline"),
    "bvp-supervised": ("josiah", _josiah_bvp("supervised"), "cross-subject-bvp", "project-baseline"),
    "bvp-simclr-trivial": ("josiah", _josiah_bvp("simclr-trivial"), "cross-subject-bvp", "project-baseline"),
    "bvp-simclr-handcrafted": ("josiah", _josiah_bvp("simclr-handcrafted"), "cross-subject-bvp", "project-baseline"),
    "autofi": ("josiah", _josiah_autofi(), "sensefi-bvp", "published-baseline"),
    "mae": ("josiah", _josiah_mae(), "cross-subject-bvp", "published-baseline"),
    "capc": ("josiah", _josiah_capc("home", "home"), "signfi-home", "published-baseline"),
    "capc-lab-to-home": ("josiah", _josiah_capc("lab", "home"), "signfi-lab-to-home", "published-baseline"),
    "autofi-uthar": ("josiah", _josiah_autofi_uthar(), "ut-har", "published-baseline"),
    "mae-uthar": ("josiah", _josiah_mae_uthar(), "ut-har", "published-baseline"),
    "doppler": ("george", _george("doppler"), "cross-subject", "proposed-method"),
    "static-perturb": ("chigozie", _chigozie("simclr-static-perturb"), "cross-environment", "proposed-method"),
    "coherent-mask": ("ihunanya", _ihunanya("simclr-coherent-mask"), "cross-subject", "proposed-method"),
    "composability-doppler": ("victor", _victor("simclr-doppler"), "cross-subject", "proposed-method"),
    "composability-coherent": ("victor", _victor("simclr-coherent-mask"), "cross-subject", "proposed-method"),
    "composability-combined": ("victor", _victor("simclr-combined"), "cross-subject", "proposed-method"),
}

BVP_METHODS = {"bvp-supervised", "bvp-simclr-trivial", "bvp-simclr-handcrafted", "autofi", "mae"}
SIGNFI_METHODS = {"capc", "capc-lab-to-home"}
UT_HAR_METHODS = {"autofi-uthar", "mae-uthar"}


def run_method(
    method: str,
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    data_root: str,
    cache_dir: str,
    representation: str,
    time_steps: int,
    max_files: int | None,
    results_root: str,
    test_dates: list[str] | None,
    bvp_root: str = "data/widar3/Widardata",
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    owner, fn, split, method_kind = METHODS[method]
    is_bvp = method in BVP_METHODS
    is_signfi = method in SIGNFI_METHODS
    is_ut_har = method in UT_HAR_METHODS
    date = datetime.now().strftime("%Y-%m-%d")
    result_dirs: list[Path] = []
    for seed in seeds:
        if is_ut_har:
            common = {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "cache_dir": cache_dir,
                "ut_har_root": data_root,
            }
        elif is_signfi:
            common = {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "cache_dir": cache_dir,
                "signfi_root": data_root,
            }
        elif is_bvp:
            common = {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "cache_dir": cache_dir,
                "bvp_root": bvp_root,
            }
        else:
            common = {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "real": True,
                "data_root": data_root,
                "cache_dir": cache_dir,
                "representation": representation,
                "time_steps": time_steps,
                "max_files": max_files,
            }
            if owner == "chigozie":
                common["test_dates"] = test_dates or ["20181128"]
        acc = fn(**common)
        if is_ut_har:
            audit = {
                "dataset": "ut-har",
                "num_classes": 7,
                "train_n": 3977,
                "val_n": 496,
                "test_n": 500,
                "input_shape": [1, 250, 90],
            }
        elif is_signfi:
            audit = {
                "dataset": "signfi",
                "pretrain_env": "lab" if method == "capc-lab-to-home" else "home",
                "eval_env": "home",
                "num_classes": 276,
                "instances_per_class": 10,
            }
        elif is_bvp:
            from src.slices.josiah.widar_bvp import audit_bvp_split

            if method == "autofi":
                audit = audit_bvp_split(bvp_root, split="sensefi", train=None)
            else:
                audit = audit_bvp_split(
                    bvp_root,
                    split="cross-subject",
                    train=None,
                    gesture_filter=[1, 2, 3, 4, 5, 6],
                )
        else:
            audit = audit_widar3_split(
                data_root,
                split=split,  # type: ignore[arg-type]
                train=None,
                test_dates=test_dates,
            )
        result_dir = Path(results_root) / f"{date}-{owner}-{method}-seed{seed}"
        if is_ut_har:
            chance = 1.0 / 7.0
            representation_tag = "ut-har-csi-amplitude"
            time_steps_tag = "ut-har-1x250x90"
        elif is_signfi:
            chance = 1.0 / 276.0
            representation_tag = "signfi-csi-amplitude"
            time_steps_tag = "signfi-3x30x200"
        elif is_bvp and method == "autofi":
            chance = 1.0 / 22.0
            representation_tag = "bvp"
            time_steps_tag = "bvp-22x20x20"
        elif is_bvp:
            chance = 1.0 / 6.0
            representation_tag = "bvp"
            time_steps_tag = "bvp-22x20x20"
        else:
            chance = 1.0 / 6.0
            representation_tag = representation
            time_steps_tag = time_steps
        write_result_bundle(
            result_dir,
            config={
                "owner": owner,
                "method": method,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "representation": representation_tag,
                "time_steps": time_steps_tag,
                "max_files": None if (is_bvp or is_signfi or is_ut_har) else max_files,
                "split": split,
                "test_dates": test_dates,
                "bvp_root": bvp_root if is_bvp else None,
                "method_kind": method_kind,
                "counts_as_published_reproduction": method in ("autofi", "capc", "capc-lab-to-home", "mae", "autofi-uthar"),
            },
            metrics={"accuracy": acc, "chance": chance},
            notes=(
                f"# {method} seed {seed}\n\n"
                "Canonical production-run artifact per docs/09-execution-roadmap.md. "
                "Published-baseline reproduction classification (exact / "
                "hardware-limited / failed) is computed at aggregate time via "
                "src.production.classify_reproduction."
            ),
            split_audit=audit,
        )
        result_dirs.append(result_dir)

    agg = aggregate_metric(result_dirs, "accuracy")
    aggregate_dir = Path(results_root) / f"{date}-{owner}-{method}-aggregate"
    if is_ut_har:
        agg_audit = audit
    elif is_signfi:
        agg_audit = audit
    elif is_bvp:
        from src.slices.josiah.widar_bvp import audit_bvp_split

        if method == "autofi":
            agg_audit = audit_bvp_split(bvp_root, split="sensefi", train=None)
        else:
            agg_audit = audit_bvp_split(
                bvp_root,
                split="cross-subject",
                train=None,
                gesture_filter=[1, 2, 3, 4, 5, 6],
            )
    else:
        agg_audit = audit_widar3_split(
            data_root, split=split, train=None, test_dates=test_dates
        )  # type: ignore[arg-type]
    write_result_bundle(
        aggregate_dir,
        config={
            "owner": owner,
            "method": method,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "representation": representation_tag,
            "time_steps": time_steps_tag,
            "max_files": None if (is_bvp or is_signfi or is_ut_har) else max_files,
            "split": split,
            "test_dates": test_dates,
            "bvp_root": bvp_root if is_bvp else None,
            "method_kind": method_kind,
            "counts_as_published_reproduction": method in ("autofi", "capc", "capc-lab-to-home", "mae", "autofi-uthar"),
        },
        metrics=agg,
        notes=f"# {method} aggregate\n\nMean/std over saved seed runs.",
        split_audit=agg_audit,
    )
    return {"seed_dirs": [str(p) for p in result_dirs], "aggregate_dir": str(aggregate_dir), "aggregate": agg}


def _parse_int_csv(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical raw-CSI production methods.")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--seeds", default="42,1337,2024")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/widar3/raw")
    parser.add_argument("--cache-dir", default="data/widar3/cache")
    parser.add_argument("--representation", choices=["real-imag", "magnitude"], default="real-imag")
    parser.add_argument("--time-steps", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--test-date", action="append", dest="test_dates", default=None)
    parser.add_argument("--bvp-root", default="data/widar3/Widardata")
    parser.add_argument("--signfi-root", default="data")
    parser.add_argument("--ut-har-root", default="data/ut_har/UT_HAR")
    args = parser.parse_args()
    if args.method in SIGNFI_METHODS:
        effective_data_root = args.signfi_root
    elif args.method in UT_HAR_METHODS:
        effective_data_root = args.ut_har_root
    else:
        effective_data_root = args.data_root
    out = run_method(
        args.method,
        seeds=_parse_int_csv(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=effective_data_root,
        cache_dir=args.cache_dir,
        representation=args.representation,
        time_steps=args.time_steps,
        max_files=args.max_files,
        results_root=args.results_root,
        test_dates=args.test_dates,
        bvp_root=args.bvp_root,
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
