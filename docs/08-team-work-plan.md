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
| 3 | Collins Izuchukwu Okafor | `c0llinx` | Calibrated phase-noise injection | Cross-chipset results + 1-page writeup |
| 4 | Ihunanya Udodiri Ajakwe | `AjakweIU` | Coherence-aware subcarrier masking | Robustness curve + 1-page writeup |
| 5 | Josiah Ayoola Isong | `isongjosiah` | Baseline reproduction | Cross-subject baseline table + 1-page writeup |
| 6 | Victor Ikenna Kanu | `xaviwho` | Composability of two physics-informed augmentations | Interaction-effects table + 1-page writeup |

The team's final paper (covering all six slices) is project-closeout work that George coordinates after slices 1–6 finish. It may become a separate publication later (target venue TBD).

---

## 4. Slice 1 — George — Doppler-aware time warping → KICS submission

**Theme.** Stretch the time axis of a CSI sample by a random factor in `[0.7, 1.4]` to simulate the same activity performed at different speeds. Cross-subject is the natural target because subjects vary in gait speed.

**Why this slice for KICS.** Lowest-risk augmentation to implement (no decomposition, no calibration, no spectral estimation). Widar3.0 cross-subject is a public benchmark. A clean comparison between Doppler warping and a generic-augmentation baseline produces a 2-page paper with a results table.

**Tracer-bullet issues:**

1. **T1.1 — Scaffold + SimCLR end-to-end with stub data.** Tiny CNN, SimCLR loss, linear probe, 10 stub samples. Pipeline runs.
2. **T1.2 — Real Widar3.0 cross-subject loader.** Replace stub. Same code now produces non-trivial accuracy.
3. **T1.3 — Generic-augmentation baseline.** Gaussian noise + random subcarrier mask. Single seed.
4. **T1.4 — Doppler-aware time warping augmentation.** Implement; integrate; single seed.
5. **T1.5 — Sanity test.** Synthetic CSI with one known frequency; warp factor 2 shifts it by 2× (within tolerance).
6. **T1.6 — Multi-seed comparison.** Three seeds for both baseline and Doppler. Mean ± std. Paired comparison.
7. **T1.7 — KICS paper draft.** LaTeX/`IEEEtran` two-column draft per the outline below.
8. **T1.8 — KICS paper polish and submission.** PDF, references, format check, submission.

**Target venue.** KICS Fall 2026 (deadline ~September 2026). KICS Winter 2027 is the backup.

**AFK execution plan.** [`docs/slice-1-afk-plan.md`](slice-1-afk-plan.md) is the detailed plan for an autonomous Claude session to execute Slice 1 end-to-end while George is away. Includes environment setup, compute strategy, dataset fallbacks, IEEEtran boilerplate, and explicit STOP points. Only relevant if running the slice AFK; teammates writing other slices should skip it.

### KICS paper outline (calibrated against George's prior KICS Winter 2026 paper)

**Working title.** *Towards Physics-Informed Augmentation for Cross-Subject WiFi CSI Sensing: Doppler-Aware Time Warping in Self-Supervised Pre-Training.* (Drop "Towards" if results are strong.)

**Format.** 2 pages, IEEE conference (`IEEEtran`) double-column, English.

**Authors (proposed).** George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim. Same affiliations as the prior paper.

**Index Terms (6–7).** Channel state information, self-supervised learning, Doppler shift, data augmentation, domain generalization, WiFi sensing.

**Abstract (~150 words), bottleneck → observation → present → numbers:**

> *Cross-subject generalization is a known weakness of WiFi channel-state-information (CSI) sensing systems, with reported accuracy drops of XX% when test subjects differ from training subjects [cite]. Existing self-supervised learning (SSL) approaches use generic data augmentations borrowed from computer vision, which do not reflect the physical phenomena that drive cross-domain shift. We observe that activity speed scales the Doppler component of CSI approximately linearly, a structure that augmentation design has not previously exploited. We present Doppler-aware time warping, a physics-informed augmentation that stretches the time axis of a CSI sample by a random factor in [0.7, 1.4] during SimCLR pre-training. On the Widar3.0 cross-subject benchmark with a frozen-encoder linear probe, Doppler-aware time warping improves accuracy by X.X ± Y.Y% over a Gaussian-noise + random-mask baseline across three random seeds.*

**Sections:**

1. **I. Introduction** (~half page). The cross-subject generalization problem; the augmentation gap; one-paragraph contributions list.
2. **II. Methodology**, two subsections matching the prior paper's A/B pattern:
   - *II-A. Doppler scaling and the time-warp operation.* Physical relationship; warp formula; one small equation.
   - *II-B. SSL pipeline integration.* SimCLR setup; encoder backbone; linear-probe protocol. **One architecture figure (Fig. 1)** showing the pipeline with the Doppler-warp module highlighted, in the same visual style as the prior paper's Fig. 1.
3. **III. Results and Discussion** (~half page). One paragraph framing the experiment, then a Table-II-style comparison: rows are configurations (no aug / Gaussian noise / random subcarrier mask / Gaussian + mask / **Doppler (ours)**), columns are accuracy with mean ± std. Bold our row. One paragraph interpreting honestly — declarative tone, no hedging.
4. **IV. Conclusion** (~quarter page). Result; three future-work threads (other physics-informed augmentations, composability study, team paper).
5. **Acknowledgment.** George's standard block (IITP / NRF / MEST / ITRC funding lines), reused verbatim from the prior paper where the grant text matches.
6. **References.** IEEE numbered, 8–10 entries: AutoFi, CAPC, CIG-MAE, SimCLR, Widar3.0, Xu et al. SSL-for-CSI benchmark, **at least 1 of George's prior KICS/ICAIIC works** (per the prior paper's pattern), domain-generalization survey if space allows.

**Acceptance for T1.8:**

- [ ] PDF renders cleanly under `IEEEtran` two-column.
- [ ] Hard 2-page limit verified.
- [ ] Every reference checked (per [`06-using-ai-well.md`](06-using-ai-well.md)).
- [ ] Acknowledgment text matches the prior paper's wording where reused.
- [ ] Submission portal upload completed; submission ID logged in `papers/kics-george/notes.md`.

---

## 5. Slice 2 — Chigozie — Static-component perturbation → cross-environment results

**Theme.** Decompose CSI into a static component (room) and dynamic component (person). Replace the static component with one from a different sample to simulate "the same activity in a different room." Cross-environment is the natural target. Highest-risk of the four augmentations because the decomposition is non-trivial.

**Tracer-bullet issues:**

1. **T2.1 — Scaffold + SimCLR end-to-end with stub data.**
2. **T2.2 — Real Widar3.0 cross-environment loader.**
3. **T2.3 — Static/dynamic decomposition** (default: temporal lowpass at 2 Hz cutoff).
4. **T2.4 — Decomposition sanity test** (synthetic slow + fast components).
5. **T2.5 — Static-component perturbation augmentation.** Swap statics across batch.
6. **T2.6 — Generic baseline + static-perturbation comparison, single seed.**
7. **T2.7 — Multi-seed comparison.**
8. **T2.8 — Results writeup** at `papers/team/static.md`.

---

## 6. Slice 3 — Collins — Calibrated phase-noise injection → cross-chipset results

**Theme.** Different WiFi chips introduce different phase noise. Profile each chip in CSI-Bench, then synthesise "what would this CSI look like on a different chip" by injecting another chip's noise profile.

**Tracer-bullet issues:**

1. **T3.1 — Scaffold + SimCLR end-to-end with stub data.**
2. **T3.2 — Real CSI-Bench cross-chipset loader.**
3. **T3.3 — Per-chip phase-noise profiling** (default: Gaussian per subcarrier).
4. **T3.4 — Profile sanity test** (KS-test "chip A with chip B's injected noise" vs real chip B).
5. **T3.5 — Phase-noise injection augmentation.**
6. **T3.6 — Generic baseline + phase-noise comparison, single seed.**
7. **T3.7 — Multi-seed comparison.**
8. **T3.8 — Results writeup** at `papers/team/phase-noise.md`.

---

## 7. Slice 4 — Ihunanya — Coherence-aware subcarrier masking → robustness results

**Theme.** Mask blocks of contiguous subcarriers whose width matches the channel's coherence bandwidth. The robustness angle: the encoder should learn to ignore coherent blocks of dropped subcarriers, so it should tolerate subcarrier loss at test time.

**Tracer-bullet issues:**

1. **T4.1 — Scaffold + SimCLR end-to-end with stub data.**
2. **T4.2 — Real Widar3.0 loader.**
3. **T4.3 — Coherence-bandwidth estimation** (default: CIR-based delay-spread, B_c ≈ 1/(5·τ)).
4. **T4.4 — Estimation sanity test** (synthetic two-tap channel).
5. **T4.5 — Coherence-aware subcarrier-block masking augmentation.**
6. **T4.6 — Held-out subcarrier robustness study** (test-time mask-N sweep).
7. **T4.7 — Multi-seed comparison.**
8. **T4.8 — Results writeup** at `papers/team/coherence.md`.

---

## 8. Slice 5 — Josiah — Baseline reproduction → cross-subject comparison table

**Theme.** Reproduce three published or canonical baselines on **Widar3.0 cross-subject** so the team has anchored numbers directly comparable to George's Slice 1.

**Tracer-bullet issues:**

1. **T5.1 — Scaffold (supervised, no SSL) on Widar3.0 cross-subject.**
2. **T5.2 — Supervised baseline produces a published-comparable cross-subject accuracy.**
3. **T5.3 — Add SimCLR pre-training with trivial augmentation.**
4. **T5.4 — Reproduce AutoFi headline number on cross-subject.**
5. **T5.5 — Reproduce CAPC headline number on cross-subject.**
6. **T5.6 — Reproduce hand-crafted-augmentation baseline on cross-subject** (3 seeds).
7. **T5.7 — Multi-seed comparison table** at `papers/team/baselines.md`.
8. **T5.8 — Gap-analysis writeup.** Identify which (metric × dataset × split) cells are empty across baselines — these are where our work has the most to contribute.

---

## 9. Slice 6 — Victor — Composability of two physics-informed augmentations

**Theme.** Do physics-informed augmentation effects compose linearly? Tested directly using **two of the four physics-informed augmentations**, recommended pair: Doppler-aware time warping + coherence-aware subcarrier masking (orthogonal axes — time and frequency — easiest to reimplement from scratch). Both augmentations are reimplemented in this slice's directory; no dependency on slices 1 or 4.

**Tracer-bullet issues:**

1. **T6.1 — Scaffold + SimCLR end-to-end with stub data.**
2. **T6.2 — Real Widar3.0 cross-subject loader.**
3. **T6.3 — Implement two physics-informed augmentations from scratch in-slice.** Includes prerequisite coherence-bandwidth estimation. Heaviest issue — budget for it.
4. **T6.4 — Single-augmentation runs on cross-subject, single seed each.**
5. **T6.5 — Combined-augmentation run, single seed.**
6. **T6.6 — Interaction term plot.** Combined effect minus sum of individual effects. Three conditions, one bar chart.
7. **T6.7 — Multi-seed across all three conditions.**
8. **T6.8 — Results writeup** at `papers/team/composability.md`. Are physics-informed augmentation effects additive on CSI?

---

## 10. Open or in-flight items not in any slice

Wave 1 onboarding issues (#1–#6) are closed. Earlier decision and epic issues (#7–#17, #25–#32) have been closed and superseded by the slice plan above; closing comments link to their replacement tracer bullets. The historical record stays intact in the closed state.

If a question comes up that does not belong in any slice — e.g., "we need a small shared utility" — open a fresh issue with the `area:infra` label and a clear scope. Cross-slice infrastructure work happens off the slice grid.

---

## 11. The team paper (project closeout)

After all six slices land, George assembles the per-slice 1-page writeups (and the cross-subject comparison table from Slice 5) into a single team paper. This may target ICTC or a workshop — venue TBD once results are in. It is not a slice; it is closeout work.
