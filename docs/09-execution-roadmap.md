# 09 — Execution Roadmap: From Where We Are To Team-Paper Results

This document sequences the work that takes the project from its current state to **a results table** that compares physics-informed augmentations against published baselines on Widar3.0 cross-subject. Two halves:

- **Part A** — reproduce the baselines we will compare against (Slice 5).
- **Part B** — run our physics-informed augmentations against those baselines (Slices 1, 2, 4, 6; Slice 3 already done).

The final deliverable is one comparison table in `papers/team/comparison.md` and George's KICS paper (Slice 1 deliverable).

If you have not yet read [`08-team-work-plan.md`](08-team-work-plan.md), read it first. This doc is the *order* in which the work in doc 08 should happen, with concrete sequencing.

---

## 0. Snapshot of where we are right now

**On `main`:**

- Slice 3 (Collins, calibrated phase-noise injection) **complete**: all eight tracer-bullets merged. Code under `src/slices/collins/`, writeup at `papers/team/phase-noise.md`.
- Slice 1 (George, Doppler-aware time warping) **partial**: T1.1 (scaffold), T1.2 (Widar3.0 cross-subject loader), T1.3 (generic-augmentation baseline functions), T1.4 + T1.5 (Doppler augmentation + sanity test) all merged. T1.6, T1.7, T1.8 remaining.
- Conventions doc 07, work-plan doc 08, AFK plan, cross-domain-drop convention fix all merged.
- Raw Widar3.0 CSI archives downloaded to `data/widar3/raw/` (~80 GB across 15 dates, verified via CRC and a csiread smoke test).

**Not started:**

- Slice 2 (Chigozie, static-component perturbation)
- Slice 4 (Ihunanya, coherence-aware subcarrier masking)
- Slice 5 (Josiah, baseline reproduction)
- Slice 6 (Victor, composability of Doppler + coherence-mask)

**Existing slice issues** (tracer-bullets, all OPEN unless tied to a merged PR):

- Slice 1: #34–#39, #91, #92 (6 closed via PR merge, T1.6/T1.7/T1.8 still open as issues)
- Slice 2: #42–#49
- Slice 3: all 8 closed (issues + PRs both merged)
- Slice 4: #58–#65
- Slice 5: #66–#73
- Slice 6: #74–#81

---

## 1. Part A — Reproduce baselines (Slice 5, Josiah)

The team's "comparison to conventional solutions" depends on these. Without anchored baseline numbers, our augmentation results have nothing meaningful to be compared against.

### 1.1 The baseline set

Five baselines, each as a separate tracer-bullet PR:

| # | Baseline | Pretext | Encoder | Headline |
|---|---|---|---|---|
| 1 | **Supervised, no SSL** | — | Tiny CNN (the conventions-doc default) | This grounds "what does no SSL get you?" |
| 2 | **SimCLR with trivial augmentation** | Contrastive, identity / random-crop only | Same backbone | Does SSL help at all on this setup? |
| 3 | **AutoFi-style** | Geometric SSL (predict rotation / transform) | AutoFi's CNN-LSTM (or our default + an AutoFi head) | The first WiFi-specific SSL paper to beat supervised on cross-domain |
| 4 | **CAPC-style** | CPC (predict future) + Barlow Twins | CAPC's encoder (or our default) | Best published augmentation analysis for CSI |
| 5 | **Hand-crafted-aug** | SimCLR with Gaussian noise + random subcarrier mask | Same backbone | This is the baseline our physics-informed augmentations must beat in Slices 1, 2, 4, 6 |

Baselines 3 and 4 are *adaptations* of the published method to our convention-doc encoder. The point is methodological reproducibility, not bit-for-bit replication — we run their *pretext task* on our pipeline and report the cross-subject linear-probe accuracy. If the resulting number falls within ~5 pp of the published headline, the reproduction is sound; if not, document the gap and the suspected cause.

### 1.2 Sequenced tracer-bullets for Slice 5

| Tracer | Issue | What ships |
|---|---|---|
| **T5.1** | #66 | Slice scaffold (supervised, no SSL) under `src/slices/josiah/`. Tiny CNN + classifier head + cross-entropy loss, on Widar3.0 raw CSI cross-subject. End-to-end runnable on stub data. |
| **T5.2** | #67 | Supervised baseline produces a defensible cross-subject accuracy on real data. Logged under `results/<date>-josiah-supervised-3seeds/`. |
| **T5.3** | #68 | Add SimCLR pre-training stage with trivial augmentation (random crop only). Linear-probe eval. |
| **T5.4** | #69 | AutoFi-style geometric SSL pretext task adapted to raw CSI. |
| **T5.5** | #70 | CAPC-style CPC + Barlow Twins adapted to raw CSI. |
| **T5.6** | #71 | Hand-crafted augmentation baseline (Gaussian + random subcarrier mask). **3 seeds.** This is the baseline column for the team's comparison table. |
| **T5.7** | #72 | Multi-seed comparison table at `papers/team/baselines.md` — all 5 baselines × 3 seeds, mean ± std. |
| **T5.8** | #73 | Gap-analysis writeup identifying which (metric × split) cells are empty across published baselines. |

### 1.3 What "done" looks like for Part A

A markdown table at `papers/team/baselines.md` with rows = the 5 baselines and a single column = cross-subject linear-probe accuracy (mean ± std, 3 seeds). Plus a paragraph documenting where our reproduction matched published numbers and where it didn't.

This unblocks every other slice's comparison column.

---

## 2. Part B — Run physics-informed augmentations (Slices 1, 2, 4, 6)

Each augmentation slice runs its own pre-training with the augmentation in place and compares to the hand-crafted baseline (T5.6). Slice 3 (Collins) has already done this; the others follow the same template.

### 2.1 Slice 1 (George) — Doppler-aware time warping

**Remaining tracer-bullets** on top of merged T1.1–T1.5:

| Tracer | Issue | What ships |
|---|---|---|
| **T1.6** | #39 | Multi-seed comparison: Doppler vs hand-crafted baseline, 3 seeds, paired test per the convention rule in doc 07. Results at `results/<date>-george-3seed/` and mirrored to local-only `papers/kics-george/results.md`. |
| **T1.7** | #91 | KICS paper draft. LaTeX `IEEEtran` two-column source in **local-only** `papers/kics-george/`. Per the AFK plan section 11 boilerplate. PR commits only the project-tracked figure script + status pointer at `papers/team/george-kics-status.md`. |
| **T1.8** | #92 | KICS paper polish + submission staging. Submission portal upload is George's manual step, not Claude's. |

**Output deliverable:** the KICS paper PDF (local, not committed) AND a `papers/team/george-kics-status.md` pointer that surfaces its existence on main.

### 2.2 Slice 2 (Chigozie) — Static-component perturbation

The 8 tracer-bullets at issues #42–#49 are unchanged from the slice plan in doc 08 section 5. The sequencing within the slice is enforced by the issue's "Blocked by" notes.

**Critical paths:** T2.3 (decomposition method) is the gate. Once that lands, T2.4 (sanity test) and T2.5 (augmentation) follow in sequence. T2.7 produces the multi-seed comparison; T2.8 produces the 1-page writeup that drops into the team's final paper.

**Output deliverable:** cross-environment accuracy table + 1-page writeup at `papers/team/static.md`.

### 2.3 Slice 4 (Ihunanya) — Coherence-aware subcarrier masking

The 8 tracer-bullets at #58–#65, again per doc 08. Robustness study (T4.6) is the slice's distinctive contribution beyond the cross-subject comparison.

**Output deliverable:** robustness curve at `papers/team/coherence-robustness.png` + 1-page writeup at `papers/team/coherence.md`.

### 2.4 Slice 6 (Victor) — Composability

Slice 6's heaviest tracer-bullet is T6.3 (reimplement Doppler + coherence-mask in-slice for independence from Slices 1 and 4). Once T6.3 lands, T6.4 (single-aug runs) and T6.5 (combined-aug run) are mechanical.

**Output deliverable:** interaction-effects bar chart + 1-page writeup at `papers/team/composability.md`.

### 2.5 What "done" looks like for Part B

Five 1-page writeups under `papers/team/` (one per slice — Collins's `phase-noise.md` already exists; Chigozie's, Ihunanya's, Victor's, and George's kics-status are TODO). Each includes:

- The augmentation's mean ± std accuracy on the target cross-domain split.
- The hand-crafted baseline's mean ± std on the same split.
- Whether the improvement passes the convention rule from doc 07.
- A brief discussion of any surprises or convention deviations.

---

## 3. Sequencing across slices

Parallelism is possible because slices are independent (per doc 08 section 2). But two things gate everything:

1. **Slice 5 T5.6 (hand-crafted baseline, 3 seeds)** is the comparison column every other slice uses. Until this lands, slice owners can run their augmentations and log raw numbers, but the "is our augmentation better than the generic baseline?" claim cannot be made.
2. **The team paper** (project-closeout) needs all six slice writeups to be drafted. George coordinates this after all slices ship.

### 3.1 Concrete priority order

The fastest path to a complete team-paper draft:

1. **Slice 5 T5.1–T5.6** (the hand-crafted baseline alone — about 3 of the 8 T5.* bullets). Once T5.6 lands, every other slice has its comparison column. T5.4 (AutoFi) and T5.5 (CAPC) can be lower priority — they provide context for the comparison table but are not blockers.
2. **Slice 1 T1.6** (multi-seed Doppler comparison) — small if T5.6 has already produced the baseline number, since the seed runs are independent.
3. **Slice 1 T1.7 + T1.8** — KICS paper drafted and staged. This is George's personal deliverable; do not gate the team paper on it.
4. **Slices 2, 4, 6 in parallel** — each owner runs their slice's full chain. Estimated 2–3 weeks per slice if focused, longer with classwork interleaving.
5. **Slice 5 T5.4, T5.5, T5.7, T5.8** — finish off the baseline-reproduction comparison table once the augmentations have shipped.
6. **Team paper** — George assembles `papers/team/*.md` into one paper (venue: ICTC or workshop, TBD).

### 3.2 Critical-path summary

```
T5.1 → T5.2 → T5.3 → T5.6 ─┬─→ (T1.6 → T1.7 → T1.8: George's KICS)
                             ├─→ Slice 2 multi-seed → writeup
                             ├─→ Slice 4 multi-seed → writeup
                             └─→ Slice 6 multi-seed → writeup
                                                      ↓
                                     Team paper integration (George)
```

T5.4, T5.5, T5.7, T5.8 are off the critical path — they enrich the comparison table after the core results land.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Slice 5 owner is slow → blocks everyone else | The hand-crafted baseline is the only Slice 5 deliverable on the critical path. T5.1, T5.2, T5.3, T5.6 can be prioritised over T5.4/T5.5. If still slow, George can take T5.6 as a side task. |
| AutoFi / CAPC reproduction off by a lot | Document the gap in T5.4/T5.5 writeups. The cross-subject comparison column the team paper needs is T5.6 (hand-crafted), not the published-method reproductions. |
| Slice 2 / 4 / 6 owners hit a hard implementation snag | The slice's tracer-bullets are designed so the slice owner can ship an honest negative result if the augmentation underperforms. Doc 08 section 10 of the AFK plan covers this for Slice 1; the same protocol applies to other slices — comment on the relevant tracer-bullet issue and write a "negative result" 1-pager. KICS / ICTC accept negative results. |
| CSI-Bench dependency for Slice 3 Stage B never resolves | Slice 3 Stage A (already complete) covers the team paper's contribution. Stage B (cross-chipset transfer on CSI-Bench) is flagged in `papers/team/phase-noise.md` as future work. The team paper does not depend on it. |
| Compute budget exceeded | Per doc 07, every slice runs on CPU or a single GPU. The biggest individual runs are SimCLR pre-trains (~30–60 min per seed on GPU). Three seeds × five augmentations × two conditions ≈ 5–10 hours total wall-clock. Tractable on one machine. If a slice blows past this, comment on the issue. |
| The 6-PR cascade pattern (force-pushes auto-closing PRs) repeats | Convention: rebase a slice's branch onto main *immediately* after an upstream slice merges, *before* attempting your own merge. The "Auto-Close Cascade" we hit on 2026-05-14 was caused by attempting `gh pr merge` on PRs that were CONFLICTING; rebase first, then merge. |

---

## 5. Conventions to remember

These are restated here for cross-reference; they all live in `docs/07-experiment-scaffold.md`:

- Encoder: tiny CNN, ~50K parameters.
- SSL framework: SimCLR (NT-Xent loss, temperature 0.5).
- Eval: linear probe on frozen encoder.
- Seeds: 3 minimum (`[42, 1337, 2024]`).
- Reporting: mean ± std.
- "Improvement is real" rule: candidate mean > baseline mean + 1 × baseline std, positive sign.
- Cross-domain drop (post-#107): `source_acc − target_acc`, positive when target underperforms source.
- Results layout: `results/<YYYY-MM-DD>-<slice>-<short-description>/` with `config.yaml`, `git_hash.txt`, `metrics.json`, `notes.md`.

---

## 6. What "ship the team paper" looks like

When all the above lands, George does the final integration:

1. Pull every `papers/team/<slice>.md` into one paper template (likely IEEE conference format, similar to the KICS submission).
2. Build a single comparison table — rows = baselines + 4 physics-informed augmentations, columns = cross-subject acc / cross-environment acc / cross-position acc as available, mean ± std.
3. Write an integrative discussion section that ties the augmentation results to the project's physics-informed thesis (per `docs/03-the-project.md`).
4. Pick a target venue: **ICTC October 2026** (IEEE Xplore indexed, the natural next step up from KICS), or a workshop at a major conference, or the project's own writeup that lives on the repo and is not externally submitted. Decision deferred until the results are in.

---

## 7. Open coordination items

These are things the team must decide together; they are not slice work:

- **Slice owners review each other's PRs.** With Collins finished and George 60% through, there is enough work merged that other slice owners should be reading the existing code (especially `src/slices/collins/` and `src/slices/george/`) before starting their own.
- **The "Auto-Close Cascade" lesson.** After 2026-05-14, the convention is: rebase BEFORE attempting merge, never the reverse. If your PR shows as CONFLICTING, fix it locally; do not run `gh pr merge --admin` hoping it sorts itself.
- **The Slice 3 "redirect cherry-pick" pattern.** Collins used a stacked-PR workflow with redirect PRs onto main. This works but produced more PRs than necessary. New slice owners should branch directly off main and merge in sequence (the simpler pattern).
- **Class deadline alignment.** The team paper is project-closeout; the KICS paper has a specific deadline (Fall 2026, ~September). The class itself has its own deadline. George should publish a one-line calendar of these dates so all six teammates can plan.
