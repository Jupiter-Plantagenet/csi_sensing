# 09 — Execution Roadmap: From Where We Are To ICTC-Paper Results

This document sequences the work that takes the project from its current state to **a full ICTC submission**: a 6-page IEEE-Xplore-indexed paper comparing physics-informed augmentations against published baselines on Widar3.0. Two halves:

- **Part A** — reproduce the baselines we will compare against (Slice 5), *exactly* as published.
- **Part B** — run our physics-informed augmentations against those baselines (Slices 1, 2, 4, 6; Slice 3 already done).

The final deliverable is a 6-page ICTC paper at `papers/team/ictc-paper/` (local-only LaTeX, project-tracked figure scripts and writeups), supported by **five tables and one figure** at `papers/team/`:

| Artifact | What it shows | Slice owner |
|---|---|---|
| `papers/team/baselines.md` — *reproduction validation* | Reproduced number vs published headline for each baseline, with the gap noted | Slice 5 |
| `papers/team/comparison.md` — *main results* | 5 baselines + 4 physics-informed augmentations, cross-subject accuracy (mean ± std, 3 seeds) | all slices |
| `papers/team/cross-domain.md` — *generalization breakdown* | Same rows × cross-subject / cross-environment / cross-position / cross-orientation columns | all slices |
| `papers/team/label-efficiency.md` — *SSL's value proposition* | Best augs vs hand-crafted baseline across 1%, 5%, 10%, 50%, 100% label fractions | Slice 5 + Slice 1 |
| `papers/team/composability.md` — *interaction effects* | Doppler, coherence, combined; deltas vs no-aug | Slice 6 |
| `papers/team/coherence-robustness.png` (figure) | Accuracy vs subcarrier-mask fraction at test time | Slice 4 |

Per-slice 1-page writeups at `papers/team/<slice>.md` (Collins's `phase-noise.md` already exists; George's `doppler.md`, Chigozie's `static.md`, Ihunanya's `coherence.md`, Victor's `composability.md` are TODO).

If you have not yet read [`08-team-work-plan.md`](08-team-work-plan.md), read it first. This doc is the *order* in which the work in doc 08 should happen, with concrete sequencing.

---

## 0. Snapshot of where we are right now

**On `main`:**

- Slice 3 (Collins, calibrated phase-noise injection) **complete**: all eight tracer-bullets merged. Code under `src/slices/collins/`, writeup at `papers/team/phase-noise.md`.
- Slice 1 (George, Doppler-aware time warping) **partial**: T1.1 (scaffold), T1.2 (Widar3.0 cross-subject loader), T1.3 (generic-augmentation baseline functions), T1.4 + T1.5 (Doppler augmentation + sanity test) all merged. T1.6, T1.7, T1.8 remaining (repurposed — see §2.1).
- Conventions doc 07, work-plan doc 08, AFK plan, cross-domain-drop convention fix all merged.
- Raw Widar3.0 CSI archives downloaded to `data/widar3/raw/` (~80 GB across 15 dates, verified via CRC and a csiread smoke test).

**Not started:**

- Slice 2 (static-component perturbation)
- Slice 4 (coherence-aware subcarrier masking)
- Slice 5 (baseline reproduction — now in scope for full reproductions)
- Slice 6 (composability of Doppler + coherence-mask)

**Execution model.** As of 2026-05-15, George and Claude are the de-facto engineering team. All slice issues are in scope for us to execute. Per-slice ownership (in doc 08 §3) is preserved nominally so commits and PRs attribute correctly, but in practice we drive every slice's tracer-bullets to completion.

---

## 1. Part A — Reproduce baselines (Slice 5)

The team's "comparison to conventional solutions" depends on these. Without anchored baseline numbers, our augmentation results have nothing meaningful to be compared against. **Reproductions are exact: each baseline runs with its published encoder, pretext task, and training schedule, not adapted to our default encoder.**

### 1.1 The baseline set

Five baselines, each as a separate tracer-bullet PR:

| # | Baseline | Encoder | Pretext / loss | Reference |
|---|---|---|---|---|
| 1 | **Supervised, no SSL** | Tiny CNN (conventions-doc default) | Cross-entropy on labelled cross-subject | grounds "what does no SSL get you?" |
| 2 | **SimCLR with trivial augmentation** | Tiny CNN (default) | NT-Xent with random crop only | grounds "does SSL help at all?" |
| 3 | **AutoFi (exact reproduction)** | AutoFi's CNN-LSTM as published | Geometric SSL with their pretext head and schedule | Yang et al. 2023 |
| 4 | **CAPC (exact reproduction)** | CAPC's encoder as published | CPC + Barlow Twins with their schedule | Barahimi et al. 2024 |
| 5 | **Hand-crafted-aug** | Tiny CNN (default) | SimCLR with Gaussian noise + random subcarrier mask | the baseline our physics-informed augmentations must beat |

For baselines 3 and 4, success means **reproducing the published headline accuracy on Widar3.0 cross-subject within ~5 pp** (publication-quality tolerance). If we land further out, document the gap and the suspected cause (different evaluation split, undocumented hyperparameters, etc.) in the per-baseline writeup. The reproduction PR contains both our number AND the published number with a citation to the source paper's table.

### 1.2 Sequenced tracer-bullets for Slice 5

| Tracer | Issue | What ships |
|---|---|---|
| **T5.1** | #66 | Slice scaffold under `src/slices/josiah/`. Tiny CNN + classifier head + cross-entropy, on Widar3.0 raw CSI cross-subject. End-to-end runnable on stub data. |
| **T5.2** | #67 | Supervised baseline produces a defensible cross-subject accuracy on real data. 3 seeds. Logged under `results/<date>-josiah-supervised-3seeds/`. |
| **T5.3** | #68 | SimCLR pre-training with trivial augmentation (random crop only) on the default encoder. 3 seeds. |
| **T5.4** | #69 | **Exact AutoFi reproduction** — implement their CNN-LSTM encoder, their geometric pretext, their training schedule. 3 seeds on cross-subject. Compare reproduced vs published headline. |
| **T5.5** | #70 | **Exact CAPC reproduction** — implement their encoder, their CPC + Barlow Twins setup, their schedule. 3 seeds. Compare reproduced vs published. |
| **T5.6** | #71 | Hand-crafted augmentation baseline (Gaussian + random subcarrier mask) on the default encoder. **3 seeds.** This is the comparison-column row used by Slices 1, 2, 4, 6. |
| **T5.7** | #72 | Build `papers/team/baselines.md` — reproduction-validation table (reproduced number, published headline, gap, citation) for all 5 baselines × 3 seeds. |
| **T5.8** | #73 | Gap-analysis paragraph identifying which (metric × split) cells are empty across published baselines — these are where our work has the most to contribute. |

### 1.3 What "done" looks like for Part A

Two artifacts on main:

1. `papers/team/baselines.md` — the reproduction-validation table.
2. The data backing `results/` directories — one per baseline × 3 seeds.

This unblocks every other slice's comparison column. T5.6's number is the row every Part-B slice compares against.

---

## 2. Part B — Run physics-informed augmentations (Slices 1, 2, 4, 6)

Each augmentation slice runs its own pre-training with the augmentation in place and compares to the hand-crafted baseline (T5.6). Slice 3 (Collins) has already done this; the others follow the same template.

### 2.1 Slice 1 — Doppler-aware time warping

**Remaining tracer-bullets** on top of merged T1.1–T1.5. T1.7 and T1.8 are repurposed: the project's no-longer aiming at KICS, so the 2-page-IEEEtran constraint is dropped. Slice 1 now ships richer results into the team paper.

| Tracer | Issue | What ships (repurposed) |
|---|---|---|
| **T1.6** | #39 | Multi-seed comparison: Doppler vs hand-crafted baseline, 3 seeds, paired test per the convention rule in doc 07. Results at `results/<date>-george-3seed/`. |
| **T1.7** | #91 | **Richer Doppler results:** sweep warp-range `[a, b]` (e.g. `[0.7, 1.4]` vs `[0.5, 1.7]`), and sweep label fractions {1, 5, 10, 50, 100}%. 3 seeds each. Output: `results/<date>-george-doppler-sweep/`. |
| **T1.8** | #92 | **1-page writeup at `papers/team/doppler.md`** — accuracy table, label-efficiency curve, narrative. Contribution to the team paper. |

The old `papers/kics-george/` is no longer the deliverable. It stays gitignored as future-work; do not invest more time in IEEEtran-2-page polish.

**Output deliverable:** `papers/team/doppler.md` + supporting `results/` directories.

### 2.2 Slice 2 — Static-component perturbation

The 8 tracer-bullets at issues #42–#49 are unchanged from the slice plan in doc 08 section 5. The sequencing within the slice is enforced by the issue's "Blocked by" notes.

**Critical path:** T2.3 (decomposition method) is the gate. Once that lands, T2.4 (sanity test) and T2.5 (augmentation) follow in sequence. T2.7 produces the multi-seed comparison; T2.8 produces `papers/team/static.md`.

### 2.3 Slice 4 — Coherence-aware subcarrier masking

The 8 tracer-bullets at #58–#65. Robustness study (T4.6) is the slice's distinctive contribution — the only one of the four augmentations that gets its own dedicated figure (`coherence-robustness.png`) in the team paper.

### 2.4 Slice 6 — Composability of Doppler + coherence-mask

Slice 6's heaviest tracer-bullet is T6.3 (reimplement Doppler + coherence-mask in-slice for independence from Slices 1 and 4). Once T6.3 lands, T6.4 (single-aug runs) and T6.5 (combined-aug run) are mechanical.

**Output deliverable:** `papers/team/composability.md` + interaction-effects bar chart at `papers/team/composability-interaction.png`.

### 2.5 What "done" looks like for Part B

Five 1-page writeups at `papers/team/` (Collins's `phase-noise.md` already shipped; George's `doppler.md`, Chigozie's `static.md`, Ihunanya's `coherence.md`, Victor's `composability.md` are TODO). Each includes the augmentation's accuracy, the hand-crafted baseline's accuracy, whether the improvement passes the convention rule, and brief discussion.

Plus four tables/figures listed in the doc's intro — built incrementally as slices ship.

---

## 3. Sequencing

Two things gate everything:

1. **T5.6 (hand-crafted baseline, 3 seeds)** is the comparison column every other slice uses. Until this lands, slice owners can run their augmentations and log raw numbers, but the "is our augmentation better than the generic baseline?" claim cannot be made.
2. **The team paper** (ICTC submission) needs all the per-slice writeups plus the five tables drafted. George coordinates assembly after all slices ship.

### 3.1 Concrete priority order

The fastest path to a complete ICTC draft:

1. **Slice 5 T5.1–T5.3 + T5.6** (the four tracer-bullets on the critical path). T5.6 is the comparison column for everyone. T5.4 / T5.5 (full AutoFi / CAPC reproductions) are higher-rigour but lower-urgency — they enrich the baseline table without being blockers.
2. **Slice 1 T1.6** (multi-seed Doppler comparison) — small if T5.6 has already produced the baseline number, since the seed runs are independent.
3. **Slices 2, 4, 6 in parallel** — each owner runs their slice's full chain. Estimated 1–2 weeks per slice with focused effort (George + Claude bandwidth, less affected by classwork than the original distributed-team model).
4. **Slice 1 T1.7 + T1.8** — richer Doppler results and `papers/team/doppler.md` writeup.
5. **Slice 5 T5.4, T5.5, T5.7, T5.8** — exact AutoFi/CAPC reproductions and the baseline table.
6. **Team paper at `papers/team/ictc-paper/`** — George assembles the five tables + per-slice writeups into the 6-page ICTC submission.

### 3.2 Critical-path summary

```
T5.1 → T5.2 → T5.3 → T5.6 ─┬─→ T1.6 → T1.7 → T1.8 (Slice 1 writeup → papers/team/doppler.md)
                             ├─→ Slice 2 multi-seed → papers/team/static.md
                             ├─→ Slice 4 multi-seed + robustness study → papers/team/coherence.md
                             └─→ Slice 6 multi-seed + interaction plot → papers/team/composability.md
                                                                                      ↓
                              T5.4, T5.5, T5.7, T5.8 → papers/team/baselines.md
                                                                                      ↓
                              George assembles papers/team/ictc-paper/ (6-page ICTC submission)
```

### 3.3 Target dates (working backwards from ICTC)

- **ICTC 2026 submission deadline:** typically July (TBC each year). Target: have the paper in shape by July 1.
- **July 1:** ICTC submission.
- **June 15:** internal review of the team-paper draft.
- **June 1:** all slice writeups landed; all five tables built.
- **May 15:** T5.6 (hand-crafted baseline) and T1.6 (Slice 1 multi-seed) shipped. Slice 2 / 4 / 6 augmentation implementations underway.
- **Now (May 15):** T5.1 scaffolding begins; doc 09 published.

If ICTC's actual deadline lands later than July, the cushion benefits everyone. If earlier, the team paper still ships in some form — even if we miss ICTC, the work lives on the repo and can target a workshop or the next conference cycle.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Exact AutoFi / CAPC reproduction lands far from published numbers | Document the gap honestly in `papers/team/baselines.md`. T5.6 (hand-crafted) is the operative baseline for the project's contribution claim; T5.4/T5.5 reproductions are validation, not the headline. |
| Slice 2 / 4 / 6 augmentation underperforms the hand-crafted baseline | Negative results are publishable. Write up honestly; flag in the slice's writeup. The team paper's framing can shift to "physics-informed augmentations are an honest, falsifiable design space" rather than "they all win." |
| CSI-Bench dependency for Slice 3 Stage B never resolves | Slice 3 Stage A (already complete) covers the team paper's contribution. Stage B (cross-chipset transfer on CSI-Bench) is flagged in `papers/team/phase-noise.md` as future work. |
| Compute budget exceeded | Per doc 07, every slice runs on CPU or a single GPU. Estimated 5–15 hours of SimCLR pre-training total across all slices (3 seeds × 5–6 conditions × 30–60 min each on GPU). Tractable on one machine. |
| Re-running into the "Auto-Close Cascade" pattern | Convention: rebase each slice's branch onto main *before* attempting merge. The 2026-05-14 cascade was caused by `gh pr merge --admin` on CONFLICTING PRs; that pattern is now off-limits. |
| ICTC deadline missed | The work lives on the repo and is reproducible. Backup venues: a workshop at a major conference, a journal special issue, or the next ICTC cycle. The team paper is not destroyed by missing one venue. |

---

## 5. Conventions to remember

These all live in `docs/07-experiment-scaffold.md`:

- Encoder: tiny CNN, ~50K parameters (for our augmentations and the hand-crafted baseline). AutoFi and CAPC reproductions use their own published encoders per §1.1.
- SSL framework: SimCLR for our augmentations. AutoFi / CAPC reproductions use their own pretext objectives.
- Eval: linear probe on frozen encoder.
- Seeds: 3 minimum (`[42, 1337, 2024]`).
- Reporting: mean ± std.
- "Improvement is real" rule: candidate mean > baseline mean + 1 × baseline std, positive sign.
- Cross-domain drop (post-#107): `source_acc − target_acc`, positive when target underperforms source.
- Results layout: `results/<YYYY-MM-DD>-<slice>-<short-description>/` with `config.yaml`, `git_hash.txt`, `metrics.json`, `notes.md`.

---

## 6. What "ship the ICTC paper" looks like

When all the above lands, George (with Claude's drafting support) does the final integration:

1. Create `papers/team/ictc-paper/main.tex` (local-only — like the KICS paper before it, the .tex source stays out of git; project-tracked artifacts are the figure scripts and the per-slice 1-page writeups under `papers/team/`).
2. Pull every `papers/team/<slice>.md` into the methodology and results sections.
3. Build the five tables (`baselines.md`, `comparison.md`, `cross-domain.md`, `label-efficiency.md`, `composability.md`) into LaTeX `tabular` form.
4. Embed `coherence-robustness.png` as the paper's main figure (alongside the SSL pipeline architecture diagram).
5. Write the integrative discussion that ties the augmentation results to the project's physics-informed thesis (per `docs/03-the-project.md`).
6. Reference list: AutoFi, CAPC, CIG-MAE, SimCLR, Widar3.0, Xu et al. SSL benchmark, plus 1–2 of George's prior KICS / ICAIIC works to anchor the research line.
7. Submit to ICTC by the published deadline.

---

## 7. Open coordination items

- **Class deadline alignment.** The project has a class deadline (TBC); the ICTC paper has its own. Confirm both on the team chat. If the class deadline lands before ICTC, the project may need to ship a class-only writeup ahead of the ICTC version.
- **Slice ownership formally.** Per-slice attribution stays as in doc 08 (Chigozie / Collins / Ihunanya / Josiah / Victor), but the engineering execution model has shifted: George + Claude drive the work, slice owners are kept informed and named on the paper. Communicate this directly to the team.
- **What to do with `papers/kics-george/`.** Currently gitignored, contains some scaffolding for the now-defunct KICS submission. Either delete locally or keep as a stretch goal (parallel 2-page KICS Fall 2026 submission carved out of the ICTC paper's Slice 1 contribution). Defer.
