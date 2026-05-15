# Proposed augmentations — BVP-reframed hypotheses

The team's literature survey (slide 17) positions our work as
**physics-grounded augmentation design for SSL on CSI**. Each slice
proposes a different augmentation derived from a named physical cause of
domain shift. The five slices were originally designed against
**Widar3.0 raw CSI cross-subject** (per the older
`docs/09-execution-roadmap.md` §1.3). That substrate is broken in our
pipeline.

This doc records the substrate pivot, the evidence that drove it, and
the five reframed BVP hypotheses we test in its place.

## Evidence that raw-CSI cross-subject fails Gate 1

| Pipeline                                           | Top-1  | Source                                                   |
|----------------------------------------------------|-------:|----------------------------------------------------------|
| Supervised TinyCNN, receivers=`[1]`                | 0.158  | `results/gate1-supervised-r1-T200-seed42.log`             |
| Supervised TinyCNN, receivers=`[1,2,3]`            | 0.169  | `results/gate1-supervised-r123-T200-seed42.log`           |
| Supervised TinyCNN, receivers=`[1..6]`             | 0.168  | `results/gate1-supervised-r123456-T200-seed42.log`        |
| SimCLR + Doppler aug (Slice 1, original)           | 0.162  | `results/2026-05-15-cross-subject-floor-finding.md`       |
| SimCLR + handcrafted (Slice 5)                     | 0.175  | same                                                     |
| AutoFi-adapted                                     | 0.174  | same                                                     |
| CAPC-adapted                                       | 0.171  | same                                                     |

Chance is **0.167** for 6 classes. Every pipeline lands within ±0.04 of
chance. The result is method-independent: four very different SSL
architectures (SimCLR, AutoFi GSS, CAPC CPC+BT, supervised end-to-end)
converge at the same floor. The bottleneck is in the substrate —
specifically the preprocessing → split combination — not the SSL
methodology, the encoder capacity, or the augmentation choice.

Diagnosis: train users 5–17 are highly imbalanced (mostly 30 samples
per user) while test users 1–4 each have hundreds of samples; the raw
CSI from one user's body geometry is a near-fingerprint at the
receiver, so cross-subject transfer is fundamentally hard at this data
scale and preprocessing depth.

## Evidence that BVP cross-subject works

| Pipeline                                  | Top-1            | Source                                                       |
|-------------------------------------------|------------------:|--------------------------------------------------------------|
| Supervised CNN on BVP                     | 0.590 ± 0.009    | `results/2026-05-15-josiah-bvp-supervised-aggregate/`         |
| SimCLR with random temporal crop          | 0.450 ± 0.018    | `results/2026-05-15-josiah-bvp-simclr-trivial-aggregate/`     |
| SimCLR with Gaussian + temporal mask      | 0.482 ± 0.005    | `results/2026-05-15-josiah-bvp-simclr-handcrafted-aggregate/` |
| MAE (BVP-adapted)                         | 0.629 ± 0.010    | `results/2026-05-15-josiah-mae-aggregate/`                    |

BVP cross-subject (train users 5–17, test users 1–4, gestures 1–6) leaves
**0.41 points of headroom over chance** with real cross-subject domain
shift. `bvp-simclr-handcrafted = 0.482 ± 0.005` is the comparison column
for proposed methods.

## What changed and why

Per-slice, we keep the **physics-grounded design principle** but reframe
each augmentation to operate on BVP, which is itself a physical
representation (body-coordinate velocity profile, derived from
multi-receiver CSI via the Widar3.0 offline pipeline).

| Slice | Original hypothesis (raw CSI) | BVP-reframed hypothesis | Implementation |
|---|---|---|---|
| **1 George — Doppler-aware time warp** | Activity speed scales the CSI Doppler component linearly; warping the time axis simulates the same gesture at a different speed. → speed invariance | **Unchanged.** BVP literally is body-coordinate velocity; warping the time axis still scales gesture speed in velocity space. The function is shape-agnostic over `(T, X, Y)`. | `src/slices/george/augmentations.py::doppler_warp`, wired as method `bvp-doppler`. |
| **2 Chigozie — Static-component perturbation** | The time-mean CSI captures slow room/multipath structure (walls, furniture); swapping it across batch teaches the encoder to ignore environment. → environment invariance | "Time-mean velocity-profile swap." On BVP, the time-mean is the *baseline velocity grid* — the part of the velocity profile that doesn't change during the gesture. Swapping it across batch teaches the encoder to ignore baseline-velocity distribution. The decomposition code (`static_dynamic_split` time-mean variant) operates correctly on any `(B, T, X, Y)` tensor. | `src/slices/chigozie/augmentations.py::static_perturb`, will be wired as `bvp-static-perturb`. |
| **3 Collins — Calibrated phase noise injection** | Different chipsets have different phase distortions; injecting samples from a fitted phase-noise profile teaches the encoder to be invariant to chipset. → chipset invariance | **REPLACED.** BVP is real-valued; phase doesn't exist on it. The substrate-compatible analog: BVP coordinate frames sometimes mis-align with true body coordinates (imperfect torso-orientation estimation; receiver geometry approximations). Apply a small random affine (rotation + translation) to the `(vx, vy)` plane. → BVP coordinate-frame invariance. | NEW: `src/slices/collins/bvp_velocity_jitter.py` (will write). |
| **4 Ihunanya — Coherence-aware subcarrier mask** | Frequency-selective fading drops a coherence-bandwidth-sized block of subcarriers; masking that block teaches the encoder to be robust. → freq-fading invariance | "Coherent velocity-band mask." On BVP, the analog is a contiguous range of `vx` (or `vy`) cells, simulating truncated sensing range or a velocity-domain occlusion. The function `coherent_block_mask` operates on axis −2 of a `(B, T, X, Y)` tensor; on BVP this is `vx`. | `src/slices/ihunanya/augmentations.py::coherent_block_mask`, will be wired as `bvp-coherent-mask`. |
| **6 Victor — Doppler + coherence composability** | Same physics as 1 + 4, applied sequentially to the same view. | Same composition logic on BVP: `doppler_warp` (time stretch) then coherent velocity-band mask. | `src/slices/victor/augmentations.py::doppler_then_coherent_mask`, will be wired as `bvp-doppler-coherent`. |

## What this changes about the team paper's argument

**Unchanged:**

- The slide-17 differentiation against AutoFi (borrowed augmentation),
  CAPC (UL/DL physics), CLAR (learned), CIG-MAE (abandoned), and SSLCSI
  (controlled-variable). Our position is still *physics-grounded
  augmentation design* against those rivals.
- Each augmentation is still a falsifiable hypothesis tied to a named
  physical cause.
- The comparison protocol: each proposed method runs against the same
  `bvp-simclr-handcrafted` baseline on the same audited cross-subject
  BVP split, 3 seeds. Improvement rule: candidate mean exceeds
  `0.482 + 1 × 0.005 = 0.487`.

**Changed:**

- The "physical cause of domain shift" each slice tests has shifted from
  CSI-domain phenomena (multipath, chipset, fading) to BVP-domain
  phenomena (baseline velocity, coordinate-frame, velocity-band
  occlusion). The design principle is identical; the named physical
  cause is different.
- Slice 3 in particular swaps "calibrated phase noise" for
  "velocity-coordinate-frame jitter." The original phase-noise
  augmentation has no BVP analog (BVP is real-valued). Stage B of
  Slice 3 (cross-chipset on CSI-Bench) remains the path to test the
  original chipset-invariance hypothesis.

## What we report in the paper

Each slice's writeup carries:

1. Original hypothesis (the raw-CSI physical cause).
2. The BVP-reframed hypothesis, with the BVP-domain physical cause.
3. Both numbers when both can be obtained: the BVP-substrate number
   (this doc's scope) and the raw-CSI-substrate number (which sits at
   chance; recorded as a failed-substrate diagnostic, not a result).
4. The comparison vs `bvp-simclr-handcrafted = 0.482 ± 0.005`.

Honesty: where the BVP analog is a true substrate-compatible analog
(Slices 1, 2, 4, 6), say so. Where it is a *substitute* hypothesis on a
different physical quantity (Slice 3: coordinate-frame jitter, not
chipset noise), say that too.
