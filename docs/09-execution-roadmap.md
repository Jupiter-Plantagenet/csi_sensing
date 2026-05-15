# 09 - Single Execution Plan: Baselines, Benchmarks, and Proposed CSI Augmentations

This is the single source of truth for the project execution plan. It subsumes
the old team work plan and the baseline reproduction protocol.

The project has two goals:

1. **Baselines and benchmark reproduction.** Reproduce the comparison methods
   we cite, especially published baselines, as exactly as possible.
2. **Proposed solution implementation and investigation.** Implement and test
   physics-informed CSI augmentations against a trustworthy hand-crafted
   baseline.

Nothing is paper-ready until the relevant acceptance gate in this document is
met.

---

## 1. Non-Negotiable Rules

### 1.1 Published Baselines

Published baselines must be reproduced as exactly as possible.

- Exact reproduction tolerance: **0.1 percentage point** absolute accuracy gap
  from the published headline value (`0.001` fractional accuracy).
- Exact reproduction means matching the paper's preprocessing, split, encoder,
  objective, optimizer, schedule, evaluation protocol, and metric.
- If exact reproduction is impossible on this hardware or with available data,
  the result is labelled **hardware-limited**, not exact.
- If any encoder, preprocessing, split, schedule, receiver set, modality, or
  fallback augmentation differs from the paper, the result is **not** a
  published-baseline reproduction.
- Do not keep runnable approximations of published baselines in baseline code.
  Approximate AutoFi/CAPC code was removed for this reason.

Every published-baseline result folder must include:

- `config.yaml`
- `metrics.json` with our value, published value, citation/table reference, and
  reproduction classification
- `git_hash.txt`
- `notes.md` explicitly stating `exact`, `hardware-limited`, or `failed`
- `split_audit.json` when using local Widar3.0 data

### 1.2 Project Baselines and Proposed Methods

Project baselines and proposed augmentations use the shared project protocol:

- Seeds: `[42, 1337, 2024]`
- Reporting: mean plus standard deviation
- Primary evaluation: frozen-encoder linear probe unless the published baseline
  requires something else
- Improvement rule: candidate mean must exceed `baseline mean + 1 * baseline std`
- Result folders must include `config.yaml`, `metrics.json`, `git_hash.txt`,
  `notes.md`, and split audit metadata where applicable

### 1.3 Production Defaults (BVP, after 2026-05-15 pivot)

The canonical input representation for project baselines and published-baseline
reproductions is **Widar3.0 BVP**, not raw CSI. Raw CSI cross-subject sat at
chance for receivers `[1]`, `[1,2,3]`, and `[1..6]` under the 2026-05-15 Gate 1
sweep (see `results/2026-05-15-cross-subject-floor-finding.md` and the
`results/2026-05-15-josiah-supervised-seed*` aggregates). BVP is the
representation the AutoFi paper and the SenseFi benchmark both use; it makes
the baseline figure reproducible.

For project-owned BVP runs:

- Dataset: Widar3.0 BVP CSV (SenseFi release, ``Widardata/`` tree)
- Canonical cross-subject split: train users `5-17`, test users `1-4`
- Canonical class filter: gestures `1-6` (Push&Pull, Sweep, Clap, Slide,
  Draw-N(H), Draw-O(H))
- Sample shape: `(T=22, vx=20, vy=20)` normalized
  `x = (x - 0.0025) / 0.0119`
- Loader/audit source: `src/slices/josiah/widar_bvp.py`

For published-baseline reproductions on BVP:

- AutoFi (Yang et al. 2022): SenseFi `Widardata/train` and `Widardata/test`
  folders verbatim; all 22 classes; SSL on train, linear probe trained on
  train, accuracy reported on test. Matches the authors' released
  `self_supervised.py` reference.
- CAPC (Barahimi et al. 2024): **hardware-limited** — paper headline is on
  SignFi with synchronized uplink/downlink CSI which Widar3.0 does not
  provide. See `papers/team/capc-hardware-limited.md`.

For raw-CSI runs (legacy, **chance-level**, retained only for diagnostic
purposes):

- `src/slices/widar.py` still loads raw Intel 5300 `.dat` with
  representation `real-imag` and the canonical filters in
  `results/2026-05-15-cross-subject-floor-finding.md`.
- Do not use raw CSI for paper-ready results until a working preprocessing
  path is identified (BVP regeneration, or a different dataset altogether).

If supervised sanity does not beat chance under the BVP defaults, do not run
expensive SSL production experiments; diagnose first.

---

## 2. What Counts as Done

### Goal 1: Baselines and Benchmarks

Done means:

- Supervised no-SSL, SimCLR-trivial, and hand-crafted SimCLR have 3-seed
  production results on the same audited project split.
- AutoFi and CAPC are either exact reproductions within 0.1 percentage point of
  the published value, or are explicitly marked hardware-limited/failed with
  a documented blocker.
- `papers/team/baselines-figure.md` contains source numbers, citations, gap
  analysis, and exact-vs-published status.
- `papers/team/baselines-figure.png` is generated from the committed source
  numbers.

### Goal 2: Proposed Solutions

Done means:

- Doppler-aware time warping, static-component perturbation, coherence-aware
  masking, and Doppler+coherence composability each have 3-seed results.
- Each proposed method is compared against the same hand-crafted baseline or
  the correct split-specific hand-crafted baseline.
- Each slice has a one-page writeup under `papers/team/`.
- The team figures are generated from committed source numbers.

---

## 3. Execution Gates

### Gate 0 - Repo Hygiene

Purpose: prevent old diagnostics from contaminating paper evidence.

Steps:

1. Keep adapted AutoFi/CAPC code out of runnable baseline paths.
2. Treat existing chance-level raw-CSI result folders as diagnostics only.
3. Use `src.production.classify_reproduction` for every published-baseline
   comparison.
4. Use `src.production_runner` only for project baselines and proposed methods,
   not exact AutoFi/CAPC until exact code exists.

Acceptance:

- No `--mode autofi`, `--mode capc`, `autofi-adapted`, or `capc-adapted`
  production path exists.
- Tests pass.

### Gate 1 - BVP Sanity (canonical, after 2026-05-15 pivot)

Purpose: prove the project-owned BVP pipeline is not broken.

Steps:

1. Audit the canonical cross-subject BVP split:

   ```bash
   python -m src.slices.josiah.widar_bvp --root data/widar3/Widardata --split cross-subject --train all --gestures 1,2,3,4,5,6
   ```

2. Run supervised sanity:

   ```bash
   python -m src.production_runner --method bvp-supervised --seeds 42 --epochs 50 --batch-size 64
   ```

Acceptance:

- Supervised top-1 accuracy must beat chance by at least 0.03 absolute
  accuracy (`> 0.1967` for 6 classes).
- 2026-05-15 sanity (2 epochs, single seed): top-1 = 0.5512. **Passes.**

Raw-CSI Gate 1 (legacy, kept for record): all three receiver configurations
failed at chance:

- receivers `[1]`:       supervised top-1 = 0.158 (results/gate1-supervised-r1-T200-seed42.log)
- receivers `[1,2,3]`:   supervised top-1 = 0.169 (results/gate1-supervised-r123-T200-seed42.log)
- receivers `[1..6]`:    supervised top-1 = 0.168 (results/gate1-supervised-r123456-T200-seed42.log)

### Gate 2 - Project Baseline Production (BVP)

Purpose: establish the comparison column for proposed methods.

Run, after Gate 1 passes:

```bash
python -m src.production_runner --method bvp-supervised --seeds 42,1337,2024 --epochs 50 --batch-size 64
python -m src.production_runner --method bvp-simclr-trivial --seeds 42,1337,2024 --epochs 300 --batch-size 64
python -m src.production_runner --method bvp-simclr-handcrafted --seeds 42,1337,2024 --epochs 300 --batch-size 64
```

Acceptance:

- Each aggregate result has three seed folders.
- Each folder has complete metadata.
- Hand-crafted BVP SimCLR becomes the comparison baseline for proposed methods.

### Gate 3 - Exact Published Baseline Reproduction

Purpose: reproduce AutoFi and CAPC as published, not as approximations.

For each published baseline:

1. Read the paper and extract the exact benchmark cell:
   - dataset
   - preprocessing
   - split
   - model/encoder
   - SSL objective
   - optimizer
   - schedule
   - batch size
   - evaluation protocol
   - metric
   - published value
2. Check whether the exact input modality exists locally.
3. Implement only the exact path. Do not add an approximation under the same
   method name.
4. Run one seed to verify the result is in range.
5. Run three seeds if the first seed is plausible.
6. Classify the result:
   - `exact` if gap <= 0.1 percentage point
   - `hardware-limited` if hardware/modality prevents exact reproduction
   - `failed` if exact reproduction was possible but did not match

Acceptance:

- AutoFi and CAPC have exact/hardware-limited/failed classifications.
- Any non-exact result is described honestly in the baseline figure notes.

### Gate 4 - Proposed Method Production

Purpose: test physics-informed augmentations against the established baseline.

Run after Gate 2 passes:

```bash
python -m src.production_runner --method doppler --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
python -m src.production_runner --method static-perturb --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
python -m src.production_runner --method coherent-mask --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
python -m src.production_runner --method composability-doppler --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
python -m src.production_runner --method composability-coherent --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
python -m src.production_runner --method composability-combined --seeds 42,1337,2024 --epochs 300 --batch-size 64 --representation real-imag --time-steps 200
```

Acceptance:

- Each proposed method has three seed folders and an aggregate.
- Each proposed method is compared to the relevant hand-crafted baseline.
- Improvements are labelled real only if they pass the project comparison rule.

### Gate 5 - Paper Artifacts

Purpose: convert runs into paper-ready evidence.

Create:

- `papers/team/baselines-figure.md`
- `papers/team/baselines-figure.png`
- `papers/team/comparison-figure.md`
- `papers/team/comparison-figure.png`
- `papers/team/cross-domain-figure.md`
- `papers/team/cross-domain-figure.png`
- `papers/team/label-efficiency-figure.md`
- `papers/team/label-efficiency-figure.png`
- `papers/team/coherence-robustness-figure.md`
- `papers/team/coherence-robustness-figure.png`
- `papers/team/composability-figure.md`
- `papers/team/composability-figure.png`

Create/update writeups:

- `papers/team/doppler.md`
- `papers/team/static.md`
- `papers/team/coherence.md`
- `papers/team/composability.md`
- `papers/team/baselines.md`

Acceptance:

- Every figure is generated from committed source numbers.
- Every writeup states the baseline, candidate, seeds, mean/std, and whether
  the result passes the comparison rule.
- Published-baseline cells state `exact`, `hardware-limited`, or `failed`.

---

## 4. Slice Responsibilities

The code remains split by owner directory, but this document controls the
sequence.

| Slice | Owner | Directory | Contribution | Required Output |
|---|---|---|---|---|
| 1 | George | `src/slices/george/` | Doppler-aware time warping | 3-seed Doppler vs hand-crafted baseline, label-efficiency sweep, `papers/team/doppler.md` |
| 2 | Chigozie | `src/slices/chigozie/` | Static-component perturbation | Cross-environment static perturbation result, `papers/team/static.md` |
| 3 | Collins | `src/slices/collins/` | Calibrated phase-noise injection | Stage A robustness result; Stage B cross-chipset only if CSI-Bench is available |
| 4 | Ihunanya | `src/slices/ihunanya/` | Coherence-aware subcarrier masking | Robustness sweep and `papers/team/coherence.md` |
| 5 | Josiah | `src/slices/josiah/` | Baselines | Project baselines plus exact AutoFi/CAPC reproduction work items |
| 6 | Victor | `src/slices/victor/` | Doppler + coherence composability | Interaction-effects result and `papers/team/composability.md` |

---

## 5. Tracer-Bullet Work Items

### Slice 1 - Doppler-Aware Time Warping

Done:

- T1.1 scaffold
- T1.2 real Widar3.0 cross-subject loader
- T1.3 generic augmentation baseline
- T1.4 Doppler-aware time warping
- T1.5 Doppler sanity test

Remaining:

1. **T1.6** - 3-seed Doppler vs hand-crafted baseline.
2. **T1.7** - warp-range sweep and label-fraction sweep.
3. **T1.8** - `papers/team/doppler.md`.

### Slice 2 - Static-Component Perturbation

1. **T2.1** - scaffold.
2. **T2.2** - real cross-environment loader.
3. **T2.3** - static/dynamic decomposition.
4. **T2.4** - decomposition sanity test.
5. **T2.5** - static-component perturbation augmentation.
6. **T2.6** - single-seed baseline vs static perturbation.
7. **T2.7** - 3-seed comparison.
8. **T2.8** - `papers/team/static.md`.

### Slice 3 - Calibrated Phase Noise

Stage A uses Widar3.0 and is valid as a robustness study. Stage B requires
CSI-Bench for the actual cross-chipset claim.

1. **T3.1** - scaffold.
2. **T3.2** - real Widar3.0 loader.
3. **T3.3** - phase-noise profile fitting.
4. **T3.4** - profile sanity test.
5. **T3.5** - phase-noise injection augmentation.
6. **T3.6** - single-seed baseline vs phase noise.
7. **T3.7** - 3-seed comparison.
8. **T3.8** - `papers/team/phase-noise.md`.

### Slice 4 - Coherence-Aware Masking

1. **T4.1** - scaffold.
2. **T4.2** - real cross-subject loader.
3. **T4.3** - coherence-bandwidth estimation.
4. **T4.4** - estimation sanity test.
5. **T4.5** - coherence-aware block masking.
6. **T4.6** - held-out subcarrier robustness sweep.
7. **T4.7** - 3-seed comparison.
8. **T4.8** - `papers/team/coherence.md`.

### Slice 5 - Baselines

1. **T5.1** - supervised scaffold.
2. **T5.2** - supervised project baseline, 3 seeds.
3. **T5.3** - SimCLR-trivial project baseline, 3 seeds.
4. **T5.4** - AutoFi exact reproduction. No approximation.
5. **T5.5** - CAPC exact reproduction. No approximation.
6. **T5.6** - hand-crafted SimCLR project baseline, 3 seeds.
7. **T5.7** - baseline figure and source numbers.
8. **T5.8** - gap analysis.

### Slice 6 - Composability

1. **T6.1** - scaffold.
2. **T6.2** - real cross-subject loader.
3. **T6.3** - in-slice Doppler and coherence-mask implementations.
4. **T6.4** - individual augmentation runs.
5. **T6.5** - combined augmentation run.
6. **T6.6** - interaction-term plot.
7. **T6.7** - 3-seed comparison.
8. **T6.8** - `papers/team/composability.md`.

---

## 6. Implementation Order

Use this order. Do not jump to expensive SSL runs before the gates pass.

1. Finish repo hygiene:
   - no runnable approximated AutoFi/CAPC baseline paths
   - exact reproduction protocol enforced in docs and result classification
   - focused tests passing
2. Run Gate 1 supervised sanity.
3. If Gate 1 passes, run Gate 2 project baselines.
4. In parallel, investigate exact AutoFi/CAPC requirements from the papers and
   official code if available.
5. Implement exact AutoFi/CAPC only after the exact preprocessing/encoder/metric
   are known.
6. Run Gate 4 proposed methods after the hand-crafted baseline exists.
7. Generate paper artifacts.

---

## 7. Risk Handling

| Risk | Response |
|---|---|
| Supervised raw-CSI sanity fails | Stop SSL production. Expand receivers, then all receivers, then consider paper-required BVP preprocessing. |
| AutoFi/CAPC exact reproduction cannot be implemented locally | Mark hardware-limited or modality-limited; document the blocker and do not claim exact reproduction. |
| Proposed augmentation underperforms hand-crafted baseline | Report honestly. Negative results are valid if the protocol is clean. |
| CSI-Bench is unavailable | Keep Slice 3 Stage B as future work; do not claim cross-chipset generalization. |
| Compute is too slow | Reduce only diagnostic runs. Do not reduce exact published schedules and still call them exact. |

---

## 8. Final Deliverable

The final project package is:

- exact/hardware-limited published-baseline reproduction record
- 3-seed project baseline record
- 3-seed proposed-method record
- source-number markdown files
- generated figures
- per-slice writeups
- a 6-page team paper assembled from those artifacts
