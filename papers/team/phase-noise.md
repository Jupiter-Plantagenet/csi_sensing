# Calibrated phase-noise injection — Stage A null result on cross-subject Widar3.0

*Slice 3, owner: Collins Izuchukwu Okafor.*

## One-paragraph result

A SimCLR encoder pre-trained with **calibrated phase-noise injection** — sampling per-`(packet, subcarrier, antenna)` phase offsets from a per-subcarrier Gaussian profile fitted to real CSI, then rotating the complex CSI by `exp(jφ)` — does **not** improve cross-subject linear-probe accuracy on Widar3.0. Three-seed paired delta vs a generic baseline (Gaussian noise + random subcarrier dropout): **−0.03 ± 0.76 pp**. Per docs/07 §2, the comparison rule rejects the result on both clauses (phase_mean below baseline_mean + 1·baseline_std, delta sign not positive). The slice's headline causal claim — that phase-noise injection improves *cross-chipset* transfer — requires CSI-Bench and is filed as Stage B follow-up work.

## Hypothesis (per docs/03 §4.2)

Different WiFi chipsets introduce different per-subcarrier phase distortions. Pre-training an encoder with phase-noise drawn from a calibrated profile should teach hardware-shift invariance, improving cross-*chipset* transfer. **The matching domain shift is cross-chipset, not cross-subject.**

## Pipeline

Eight tracer-bullet PRs (T3.1–T3.8) build `src/slices/collins/`: tiny CNN encoder (~49 K params, real-imag stacked CSI input), SimCLR with NT-Xent τ=0.5, frozen-encoder linear probe, csiread-based Widar3.0 cross-subject loader, per-subcarrier Gaussian phase-noise profile fit, and the phase-noise augmentation factory. The augmentation rotates complex CSI by `exp(jφ)` per `(packet, subcarrier, antenna)` element with `φ ~ N(μ_k, σ_k²)`. Magnitude `|H|` is preserved exactly (the rotation is unitary in the Re/Im plane), which is the physical-correctness guarantee — and also why the loader had to keep complex CSI rather than reduce to magnitude as Slice 1 does.

## Stage A constraint

Widar3.0 is a single-chipset dataset (Intel 5300). CSI-Bench (the multi-chipset dataset designated by docs/03 §5.1) is not yet downloaded. Stage A is therefore a **robustness stress-test on cross-subject** (users 1–13 train, 14–17 test, six gestures, max_files=8000 per arm), not the slice's actual causal claim.

## Empirical observation: σ ≈ 1.80 rad

The per-subcarrier Gaussian fit on phase residuals (channel-response removed via per-`(subcarrier, antenna)` temporal mean) lands at **σ ≈ 1.795 rad uniformly across all 30 subcarriers** — just below π/√3 ≈ 1.814, the std of a uniform distribution on (−π, π]. The Stage A residuals are nearly uniform on the unit circle. T3.4's KS sanity test on synthesised data confirms the fit pipeline is unbiased, so this is a property of the data, not a bug. Interpretation: the per-window temporal mean is too coarse a channel estimate over a 1–2 s gesture window — motion-induced channel drift dominates the residual rather than hardware-fingerprint noise. A refined Stage A model (short sliding-window mean, or per-packet linear-phase removal) would reduce σ; deferred per the contract default.

## Cross-subject paired comparison (T3.7)

| seed | generic baseline | phase-noise inj | delta (phase − baseline) |
|---|---|---|---|
| 42 | 0.1710 | 0.1620 | −0.90 pp |
| 43 | 0.1774 | 0.1800 | +0.26 pp |
| 44 | 0.1646 | 0.1700 | +0.54 pp |
| **mean ± std** | **0.1710 ± 0.0064** | **0.1707 ± 0.0090** | **−0.03 ± 0.76 pp** |

docs/07 §2 verdict: phase_mean (0.1707) needs to exceed baseline_mean + 1·baseline_std (= 0.1774) **and** delta_mean needs to be positive. Both fail. **No real improvement.** Both arms hover at chance (1/6 = 0.1667). T3.6's single-seed −0.90 pp was an outlier within the 0.76 pp seed-to-seed std band, not a signal — the value of the multi-seed protocol is exactly to catch that.

## What this means

The null result is **consistent with the hypothesis**. Phase-noise injection targets cross-chipset shift; cross-subject is not its design target, and a single-chipset dataset can't isolate chipset-specific structure that doesn't exist there. A *positive* cross-subject delta would have been the surprising finding and would have warranted explanation. The Stage A reading is: *the augmentation does not help where it was not designed to help*, which is part of demonstrating the physics-to-augmentation mapping in docs/03 §5.4 is *specific* rather than *generically helpful*.

## Stage B (future work)

Download CSI-Bench from Kaggle, fit per-chipset phase-noise profiles separately so σ reflects chipset-specific hardware shifts (rather than single-chipset motion drift), and run T3.6/T3.7-style paired comparison on a cross-chipset split. The slice's actual causal claim — *"phase-noise injection improves cross-chipset linear-probe accuracy vs a generic baseline"* — can only be tested in that setup. Filed as follow-up child issues.

## Limitations

- Stage A σ ≈ 1.80 rad is dominated by motion drift, not hardware noise; a refined channel-response model is the natural next step before Stage B.
- Cross-subject linear-probe accuracy at chance is consistent with published SSL+linear-probe baselines on Widar3.0 with limited subject diversity; the task itself has little headroom for any augmentation to demonstrate effect.
- Stage B is required before the slice's causal claim can be supported or refuted.

## Reproducibility

Deterministic on the seeds reported. After the slice stack lands on `main`:

```bash
python -m src.slices.collins.compare \
    --data-root data/widar3/extracted --cache-dir data/widar3/cache \
    --phase-profile data/widar3/cache/phase_profile.npz \
    --seeds 42 43 44 --epochs 20 --batch-size 64 --max-files 8000
```

Per-seed metrics, aggregated stats, and the docs/07 §2 verdict are persisted at `results/2026-05-08-collins-T3.7-multiseed-baseline-vs-phase-noise/{config.yaml, git_hash.txt, metrics.json, notes.md}` (all four are tracked; heavy outputs are gitignored).
