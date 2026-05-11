# T3.6 — single-seed comparison

Date: 2026-05-08
Seed: 42
Branch: `collins/T3.6-baseline-vs-phase-noise`

## Numbers

| arm | linear-probe acc | pretrain final loss |
|---|---|---|
| generic baseline | 0.1710 | 2.9294 |
| phase-noise inj  | 0.1620 | 2.9449 |
| delta (phase − baseline) | **−0.90 pp** | — |

Both arms also match the corresponding standalone runs from T3.2 (0.171) and
T3.5 (0.162) bit-for-bit; same seed → same data ordering → only the
augmentation differs, which is exactly what this comparison is supposed to
isolate.

## What was expected

Per docs/03 §4.2: phase-noise injection is the *cross-chipset* augmentation;
cross-subject is a robustness stress-test for Stage A but is **not the shift
the augmentation was designed to address**. A near-zero or slightly negative
delta would be consistent with the hypothesis. A strongly positive delta on
cross-subject would actually be surprising — it would mean phase-noise was
doing work outside its intended physical regime.

Quantitative bar: at n=8000 test samples the standard error of accuracy is
sqrt(p·(1−p)/n) ≈ 0.0042 = 0.42 pp. The −0.90 pp delta is ~2 SE — too small
to call a real effect with one seed, large enough that T3.7's three-seed
sweep should resolve it.

## What we saw

The −0.90 pp delta points in the "phase-noise injection at Stage A doesn't
help cross-subject" direction, consistent with the hypothesis. Two
interpretations remain possible until T3.7:

1. **Genuine null effect**, σ ≈ 1.80 rad scrambles phase too aggressively to
   leave any cross-subject-useful structure for the encoder.
2. **Single-seed artefact**, three-seed mean lands within ±SE of zero.

Either way, this single-seed run alone does NOT meet docs/07 §2's "real
improvement" rule — that rule is multi-seed only. T3.7 will resolve.

The slice's headline causal claim still lives at Stage B (cross-chipset on
CSI-Bench); this is the Stage A robustness probe, calibrated as expected.

## Environment

CPU-only, 8 cores, 13 train users (1–13), 4 test users (14–17), 6 gestures,
extracted Widar3.0 from CSI_2018*.zip subset. See `config.yaml`.
