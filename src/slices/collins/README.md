# Slice 3 — Collins — Calibrated phase-noise injection

This directory holds the end-to-end SSL pipeline for the calibrated
phase-noise-injection slice. See `docs/08-team-work-plan.md` section 6 for the
slice roadmap and `docs/07-experiment-scaffold.md` for project-wide defaults.

## Slice plan, in one paragraph

Different WiFi chipsets introduce different phase distortions. The slice fits a
phase-noise profile from real CSI, then synthesises "what would this CSI look
like under different phase noise" by injecting samples from the profile during
SimCLR pre-training. **Stage A** (this branch and the immediately following
ones) prototypes the augmentation on Widar3.0 raw CSI (Intel 5300, single
chipset) and reports cross-subject robustness. **Stage B** (follow-up issues
once CSI-Bench is downloaded) fits per-chipset profiles and reports cross-
chipset transfer — the slice's headline causal claim.

## Layout

| File | What it holds |
|---|---|
| `encoder.py` | `TinyCNN` (~40K params for stub, ~49K for real-imag-stacked Widar3.0) and the (T,S,A)→(C,T) reshape helper. |
| `ssl.py` | SimCLR wrapper, projection head, NT-Xent loss, pre-training loop. |
| `eval.py` | Linear-probe evaluation on a frozen encoder (sklearn LogisticRegression). |
| `data.py` | `StubCSI` (T3.1) and `Widar3CrossSubject` (T3.2). |
| `phase_profile.py` | T3.3 — per-subcarrier Gaussian fit on phase residuals. |
| `augmentations.py` | T3.5 — generic baseline + phase-noise injection. |
| `compare.py` | T3.6 / T3.7 — single- or multi-seed paired comparison driver. |
| `run.py` | End-to-end entrypoint that wires the pieces together. |

## Run

### T3.1 — stub-data smoke (no real data needed)

```bash
python -m src.slices.collins.run
pytest tests/slices/collins/
```

Both should complete without error. Stub-data accuracy is ~chance and is not
meaningful — the test only verifies the pipeline runs.

### T3.2 — real Widar3.0 cross-subject

Pre-requisite: the `CSI_2018*.zip` files have been extracted under
`data/widar3/extracted/`, producing `data/widar3/extracted/<YYYYMMDD>/user<U>/`
trees with `user<U>-<G>-<T>-<O>-<I>-r<R>.dat` files.

```bash
python -m src.slices.collins.run \
    --data-root data/widar3/extracted \
    --cache-dir data/widar3/cache \
    --epochs 20 --batch-size 64 --max-files 8000 --seed 42
```

The first run cold-parses each `.dat` via `csiread.Intel`, normalises, and
caches the result under `--cache-dir`. Subsequent runs hit the cache directly.

#### Expected smoke result

With the command above and the `make_generic_aug(sigma=0.3,
subcarrier_drop_prob=0.15)` stopgap augmentation, the linear-probe on
cross-subject test (users 14–17) reaches **~0.171** versus chance ~0.167. The
margin is small — about one standard error at n=8000 — and reflects two known
properties of the setup, not a loader bug:

- Cross-subject distribution shift on Widar3.0 is severe; even a supervised
  end-to-end CNN reaches train-acc ~0.46 but stays at chance on the test split
  with our data scope, confirming the data is informative but the encoder
  overfits to subject-specific gait patterns.
- Slice 3's *intended* augmentation is calibrated phase-noise injection, which
  targets cross-*chipset* shift, not cross-subject. The Stage A smoke is a
  stress test of robustness; the slice's causal claim is Stage B (cross-chipset
  on CSI-Bench, follow-up issues).

The supervised-baseline check is preserved as a one-off diagnostic; see the PR
description for the full numbers.

## Status

All eight Stage A tracer bullets landed:

| Step | What | PR |
|---|---|---|
| T3.1 | Scaffold + SimCLR end-to-end with stub data | #93 |
| T3.2 | Real Widar3.0 cross-subject loader (csiread + real-imag stack) | #94 |
| T3.3 | Per-subcarrier phase-noise profile fit | #95 |
| T3.4 | Profile sanity test (KS against synthetic Gaussian) | #96 |
| T3.5 | Phase-noise injection augmentation | #98 |
| T3.6 | Single-seed paired comparison | #99 |
| T3.7 | Three-seed paired comparison + docs/07 §2 verdict | #100 |
| T3.8 | 1-page writeup at `papers/team/phase-noise.md` | this PR |

**Stage A headline: null cross-subject result (delta −0.03 ± 0.76 pp across 3
seeds), consistent with the augmentation's intended target being
cross-chipset shift.** See [`papers/team/phase-noise.md`](../../../papers/team/phase-noise.md)
for the slice writeup.

**Stage B (cross-chipset on CSI-Bench) is filed as follow-up child issues**
once the dataset is downloaded; the slice's actual causal claim can only be
tested in that setup.
