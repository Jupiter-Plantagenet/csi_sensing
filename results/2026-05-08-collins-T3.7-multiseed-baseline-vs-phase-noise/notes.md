# T3.7 — three-seed comparison

Date: 2026-05-08
Seeds: [42, 43, 44]

## Per-seed numbers

| seed | generic baseline | phase-noise inj | delta (phase − baseline) |
|---|---|---|---|
| 42 | 0.1710 | 0.1620 | -0.90 pp |
| 43 | 0.1774 | 0.1800 | +0.26 pp |
| 44 | 0.1646 | 0.1700 | +0.54 pp |

## Aggregate (paired)

- generic baseline:  mean=0.1710  std=0.0064
- phase-noise inj:   mean=0.1707  std=0.0090
- delta:             mean=-0.03 pp  std=0.76 pp

## docs/07 §2 decision rule

> phase_mean > baseline_mean + 1*baseline_std AND delta_mean > 0  (requires ≥ 3 seeds)

**real improvement: NO**

## What was expected

Per docs/03 §4.2 phase-noise injection is the *cross-chipset* augmentation; cross-subject is a robustness stress-test of Stage A and is not the shift the augmentation was designed for. A near-zero or slightly negative delta is consistent with the hypothesis. The slice's headline causal claim still lives at Stage B (cross-chipset on CSI-Bench, follow-up issues).

## What we saw

A clean honest null result.

**Three-seed delta: −0.03 ± 0.76 pp.** The mean is indistinguishable from
zero relative to its own seed-to-seed std — calibrated phase-noise
injection (Stage A profile, σ ≈ 1.80 rad) neither helps nor hurts the
cross-subject linear-probe accuracy of a SimCLR-pretrained encoder on
Widar3.0 with our setup.

**Seed 42's −0.90 pp from T3.6 was an outlier** (within the 0.76 pp
seed-to-seed std), not a signal. Seeds 43 and 44 give +0.30 pp and
+0.54 pp respectively. This is the value of the multi-seed protocol.

**Both arms hover at chance** (1/6 ≈ 0.167). The standard error at
n=8000 test samples is ≈ 0.42 pp, so the cross-subject task itself is
barely above noise at this data scope. The slice's *real* causal claim
— cross-chipset on CSI-Bench — is the right place to look for an
augmentation effect; a single-chipset Widar3.0 fit can't isolate
chipset-fingerprint structure that doesn't exist there.

**The hypothesis is consistent with the result.** Per docs/03 §4.2
phase-noise injection is designed for cross-*chipset*; cross-subject
was the Stage A stress-test, not the shift this augmentation should
help. A positive cross-subject delta would have been the surprising
finding and would have warranted explanation. A null result is what
the physics predicted.

**For T3.8 / the 1-page writeup**: report Stage A as "phase-noise
injection produces a null result on cross-subject Widar3.0 (delta
−0.03 ± 0.76 pp across 3 seeds), consistent with the augmentation's
intended target being cross-*chipset* shift; Stage B (CSI-Bench) is
where the slice's causal claim should be tested." This is a useful
contribution to the team's findings table — it tells us where the
augmentation does NOT help, which is part of demonstrating the
physics-to-augmentation mapping is specific.
