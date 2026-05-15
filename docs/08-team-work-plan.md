# 08 — Team Work Plan: Six Independent Tracer-Bullet Slices

This document is the project's roadmap. Every open work item belongs to one of six **slices**, each owned by one teammate. Each slice is a chain of small **tracer-bullet** issues that produce something runnable at every step.

**No issue in any slice depends on an issue from another slice.** Concretely: any teammate can finish their slice on schedule even if another teammate is two weeks behind.

If you have not yet read [`07-experiment-scaffold.md`](07-experiment-scaffold.md), read it first. This doc is the roadmap; that doc is the convention reference.

---

## 1. Why "tracer bullets"

A tracer bullet is a complete vertical slice of the system that runs end-to-end at every stage, even when most of it is stubbed. The first tracer-bullet issue in a slice produces a pipeline that runs; it does not produce a *correct* result. Subsequent issues replace stubs with real components.

Two reasons this matters:

1. **Plumbing bugs are expensive.** A pipeline that does not run end-to-end hides them. By the time the bug surfaces, three other components depend on the broken assumption. With tracer bullets, every issue's PR proves the pipeline still runs.
2. **Independent slices need independent scaffolds.** If everyone shares one scaffold, one person's broken commit blocks the whole team. With six per-person scaffolds, breakage is localised.

The cost is some duplication. Two slice owners will each write a Widar3.0 loader; the two loaders may differ in small ways. We accept that. Owning an end-to-end pipeline beats dividing one shared codebase into surgical patches, especially for graduate students learning the field.

---

## 2. What "independent" means in practice

- Each slice lives under `src/slices/<short-owner-id>/` and is owned by one person.
- The owner can read and copy from any other slice's directory freely.
- No PR in one slice blocks a PR in another slice.
- Shared defaults — encoder choice, SSL framework, evaluation protocol — live in [`07-experiment-scaffold.md`](07-experiment-scaffold.md) as **conventions**, not enforced via blocking dependencies. A slice may deviate with a one-paragraph note in its results writeup.

---

## 3. The six slices

| Slice | Owner | GitHub | Theme | Deliverable |
|---|---|---|---|---|
| 1 | George Chidera Akor | `Jupiter-Plantagenet` | Doppler-aware time warping | KICS submission |
| 2 | Chigozie Athanasius Nnadiekwe | `Gozie01` | Static-component perturbation | Cross-environment results + 1-page writeup |
| 3 | Collins Izuchukwu Okafor | `c0llinx` | Calibrated phase-noise injection | Cross-chipset (or robustness) results + 1-page writeup |
| 4 | Ihunanya Udodiri Ajakwe | `AjakweIU` | Coherence-aware subcarrier masking | Robustness curve + 1-page writeup |
| 5 | Josiah Ayoola Isong | `isongjosiah` | Baseline reproduction | Cross-subject baseline table + 1-page writeup |
| 6 | Victor Ikenna Kanu | `xaviwho` | Composability of Doppler + coherence-aware masking | Interaction-effects table + 1-page writeup |

The team's final paper (covering all six slices) is project-closeout work that George coordinates after slices 1–6 finish. It may become a separate publication later (target venue TBD).

### A note on the data

All slices that consume Widar3.0 operate on **raw Channel State Information** from Intel 5300 NICs (the original `.dat` format). The dataset is downloaded from IEEE DataPort to `data/widar3/raw/` as 15 archives (one per recording-session date), totalling roughly 80 GB. Each `.dat` file parses via `csiread.Intel(nrxnum=3, ntxnum=1, pl_size=10)` to a complex CSI tensor of shape `(T_packets, 30 subcarriers, 3 antennas, 1)`; squeeze the last dim. Sampling rate is 1000 packets/sec; a gesture instance lasts ~1–2 s. Filename schema: `userN-G-P-O-T-rR.dat` for user-gesture-position-orientation-trial-receiver. The 15 dates span the three published Widar3.0 environments (classroom, office, hall), enabling genuine cross-environment evaluation for Slice 2.

**Slice 3 has a remaining data dependency.** Calibrated phase-noise injection across chipsets needs CSI-Bench (a different dataset, hosted on Kaggle), since Widar3.0 records on a single chipset family (Intel 5300). Slice 3 can prototype the augmentation pipeline on Widar3.0 raw CSI now and report a robustness result; the cross-chipset claim itself awaits CSI-Bench access. The slice's tracer-bullets are written to be incrementally upgradable once that data is in hand.

---

## 4. Slice 1 — George — Doppler-aware time warping → contribution to team paper

**Theme.** Stretch the time axis of a raw-CSI sample by a random factor in `[0.7, 1.4]` to simulate the same activity performed at different speeds. Cross-subject is the natural target because subjects vary in gait speed.

**Why this slice matters.** Lowest-risk augmentation to implement (no decomposition, no calibration, no spectral estimation). Widar3.0 cross-subject is a public benchmark with published baselines. Provides the project's first end-to-end physics-informed augmentation result; serves as the template for slices 2, 4, 6.

**Tracer-bullet issues:**

1. **T1.1 — Scaffold + SimCLR end-to-end with stub data.** Tiny CNN, SimCLR loss, linear probe, 10 stub CSI samples shaped `(T, 30, 3)`. Pipeline runs. **MERGED.**
2. **T1.2 — Real Widar3.0 raw-CSI cross-subject loader.** Parse `.dat` files via `csiread`, filter to canonical 6 gestures + position 1 + orientation 1 + receiver 1, build cross-subject user split. **MERGED.**
3. **T1.3 — Generic-augmentation baseline.** Gaussian noise + random subcarrier mask. Single seed. **MERGED.**
4. **T1.4 — Doppler-aware time warping.** Stretch the time axis by a factor in `[0.7, 1.4]`. Single seed. **MERGED.**
5. **T1.5 — Sanity test.** Synthetic CSI with one known dominant frequency; factor-2 warp halves the dominant frequency (within tolerance). **MERGED.**
6. **T1.6 — Multi-seed comparison.** Three seeds for both baseline and Doppler. Mean ± std. Paired comparison.
7. **T1.7 — Richer Doppler results.** Sweep warp-range and sweep label fractions (1, 5, 10, 50, 100%); 3 seeds each. Output: figures for `papers/team/label-efficiency-figure.png` (Slice 1's contribution to it).
8. **T1.8 — 1-page writeup at `papers/team/doppler.md`** describing the result, linking to the figures, contributing to the team paper.

**Target venue.** ICTC October 2026 (the team paper). The previously planned KICS Fall 2026 submission has been folded into the team paper; the 2-page IEEEtran constraint no longer applies.

---

## 5. Slice 2 — Chigozie — Static-component perturbation → cross-environment results

**Theme.** Decompose a raw-CSI sample into a **static component** (the slowly-varying contribution from walls, furniture, ceiling — i.e. the room) and a **dynamic component** (the fast-varying contribution from the moving person). Practical default: temporal lowpass filter at 2 Hz. Replace the static component with one from a different sample to simulate "the same activity in a different room." Cross-environment is the natural target. Highest-risk of the four augmentations because the decomposition is non-trivial.

**Tracer-bullet issues:**

1. **T2.1 — Scaffold + SimCLR end-to-end with stub data.**
2. **T2.2 — Real Widar3.0 raw-CSI cross-environment loader.** Build a split where train and test draw from different recording dates corresponding to different rooms (the 15 dates span 3 environments; verify the date-to-room mapping empirically during T2.2 by spot-checking a sample from each date).
3. **T2.3 — Static/dynamic decomposition** (default: temporal lowpass at 2 Hz cutoff).
4. **T2.4 — Decomposition sanity test** (synthetic slow + fast components).
5. **T2.5 — Static-component perturbation augmentation.** Swap statics across the batch.
6. **T2.6 — Generic baseline + static-perturbation comparison, single seed.**
7. **T2.7 — Multi-seed comparison.**
8. **T2.8 — Results writeup** at `papers/team/static.md`.

---

## 6. Slice 3 — Collins — Calibrated phase-noise injection → cross-chipset (or robustness) results

**Theme.** Different WiFi chips introduce different phase noise. The full claim — fitting per-chip phase-noise distributions and injecting them across chipsets — requires CSI-Bench (not Widar3.0; Widar3.0 records on Intel 5300 only). The slice has two stages:

- **Stage A (now, on Widar3.0 raw CSI):** fit a single phase-noise model from a held-out training subset of Widar3.0 raw CSI, inject it during SimCLR pre-training, and measure robustness to held-out phase perturbations on cross-subject test. Anchored augmentation but a weaker claim than cross-chipset transfer.
- **Stage B (when CSI-Bench is downloaded):** refit phase-noise models per chipset (CSI-Bench has 5 chip families), inject across chipsets, and measure cross-chipset transfer accuracy. This is the original cross-chipset claim.

**Tracer-bullet issues** are written to ship Stage A first; Stage B is added as a follow-up child issue once CSI-Bench is in hand.

1. **T3.1 — Scaffold + SimCLR end-to-end with stub raw-CSI data.**
2. **T3.2 — Real Widar3.0 raw-CSI cross-subject loader** (CSI-Bench loader filed as a follow-up issue once data arrives).
3. **T3.3 — Phase-noise model fitting** (default: independent Gaussian per subcarrier on phase residuals after channel-response removal).
4. **T3.4 — Profile sanity test** (KS-test against held-out perturbation distribution; sub-issue for the full cross-chipset KS-test once CSI-Bench is in hand).
5. **T3.5 — Phase-noise injection augmentation.**
6. **T3.6 — Generic baseline + phase-noise comparison, single seed.**
7. **T3.7 — Multi-seed comparison.**
8. **T3.8 — Results writeup** at `papers/team/phase-noise.md`. Stage A robustness numbers; Stage B framed as future work pending CSI-Bench.

---

## 7. Slice 4 — Ihunanya — Coherence-aware subcarrier masking → robustness results

**Theme.** Mask blocks of contiguous subcarriers whose width matches the channel's coherence bandwidth, simulating realistic frequency-selective fading. The robustness angle: the encoder should learn to ignore coherent blocks of dropped subcarriers, so it should tolerate subcarrier loss at test time.

**Tracer-bullet issues:**

1. **T4.1 — Scaffold + SimCLR end-to-end with stub raw-CSI data.**
2. **T4.2 — Real Widar3.0 raw-CSI loader.** (Cross-subject split is fine for robustness work.)
3. **T4.3 — Coherence-bandwidth estimation** (default: CIR-based delay-spread, B_c ≈ 1/(5·τ); compute via inverse FFT across subcarriers).
4. **T4.4 — Estimation sanity test** (synthetic two-tap channel with known delay spread).
5. **T4.5 — Coherence-aware subcarrier-block masking augmentation.**
6. **T4.6 — Held-out subcarrier robustness study** (test-time mask-N sweep over the 30 subcarriers).
7. **T4.7 — Multi-seed comparison.**
8. **T4.8 — Results writeup** at `papers/team/coherence.md`.

---

## 8. Slice 5 — Josiah — Baseline reproduction → cross-subject comparison table

**Theme.** Reproduce three published or canonical baselines on **Widar3.0 cross-subject (raw CSI)** so the team has anchored numbers directly comparable to George's Slice 1.

**Tracer-bullet issues:**

1. **T5.1 — Scaffold (supervised, no SSL) on Widar3.0 raw-CSI cross-subject.**
2. **T5.2 — Supervised baseline produces a published-comparable cross-subject accuracy.** Within ~5 pp tolerance of a published Widar3.0 cross-subject number.
3. **T5.3 — Add SimCLR pre-training with trivial augmentation.** SSL pipeline runs.
4. **T5.4 — Reproduce AutoFi headline number on cross-subject.** Geometric SSL on raw CSI.
5. **T5.5 — Reproduce CAPC headline number on cross-subject.** CPC + Barlow Twins on raw CSI.
6. **T5.6 — Reproduce hand-crafted-augmentation baseline on cross-subject** (3 seeds). SimCLR with Gaussian noise + random subcarrier mask, same configuration as George's Slice 1 — directly comparable numbers.
7. **T5.7 — Multi-seed comparison table** at `papers/team/baselines.md`.
8. **T5.8 — Gap-analysis writeup.** Identify which (metric × dataset × split) cells are empty across published baselines — these are where the team's work has the most to contribute.

---

## 9. Slice 6 — Victor — Composability of Doppler + coherence-aware masking → interaction-effects writeup

**Theme.** Do physics-informed augmentation effects compose linearly? The deck flags "isolating effects" as a difficulty. This slice tests it directly using **Doppler-aware time warping + coherence-aware subcarrier masking** — two of the four physics-informed augmentations, operating on orthogonal axes (time and frequency), both reimplemented in this slice's directory for full independence from slices 1 and 4.

**Tracer-bullet issues:**

1. **T6.1 — Scaffold + SimCLR end-to-end with stub raw-CSI data.**
2. **T6.2 — Real Widar3.0 raw-CSI cross-subject loader.**
3. **T6.3 — Implement Doppler + coherence-aware-masking from scratch in-slice.** Reimplemented here for slice independence. Includes the prerequisite coherence-bandwidth estimation. Heaviest issue — budget for it.
4. **T6.4 — Single-augmentation runs on cross-subject, single seed each.**
5. **T6.5 — Combined-augmentation run, single seed.** Decide and document the composition strategy: sequential within a view, or one per view.
6. **T6.6 — Interaction term plot.** Combined effect minus sum of individual effects. Three conditions, one bar chart.
7. **T6.7 — Multi-seed across all three conditions.**
8. **T6.8 — Results writeup.** Do physics-informed augmentation effects compose linearly on raw CSI? 1-page markdown at `papers/team/composability.md`.

---

## 10. Open or in-flight items not in any slice

Wave 1 onboarding issues (#1–#6) are closed. Earlier decision and epic issues (#7–#17, #25–#32) have been closed and superseded by the slice plan above; closing comments link to their replacement tracer bullets. The historical record stays intact in the closed state.

If a question comes up that does not belong in any slice — e.g., "we need a small shared utility" — open a fresh issue with the `area:infra` label and a clear scope. Cross-slice infrastructure work happens off the slice grid.

---

## 11. The team paper (project closeout)

After all six slices land, George assembles the per-slice 1-page writeups (and the cross-subject comparison table from Slice 5) into a single team paper. This may target ICTC or a workshop — venue TBD once results are in. It is not a slice; it is closeout work.
