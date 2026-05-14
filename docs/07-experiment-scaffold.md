# 07 — Experiment Scaffold and Project Conventions

This document describes the **defaults** every slice in [`08-team-work-plan.md`](08-team-work-plan.md) uses unless a slice explicitly opts out. Conventions are not enforced via blocking dependencies — slices remain independent. Conventions exist so that:

- numbers from different slices are directly comparable,
- code is easier to read across slices,
- the team's final paper can pull from any slice's results without per-slice translation.

If you have not yet read [`08-team-work-plan.md`](08-team-work-plan.md), read it first. This doc is reference; that doc is the roadmap.

---

## 1. Defaults at a glance

| Component | Default | Why |
|---|---|---|
| Encoder backbone | Small CNN — three conv layers + global pool, ~50K parameters. | Comparability across slices and fast iteration on a single GPU. |
| SSL framework | SimCLR (NT-Xent loss). | Canonical, simpler than Barlow Twins, audience-recognisable. |
| Evaluation protocol | Linear probe on a frozen encoder. | Cleaner causal claim about representation quality than full fine-tuning. |
| Random seeds | 3 minimum. Mean ± std reported for every headline number. | Project-wide statistical-reporting convention. |
| Cross-domain delta | `target_acc − source_acc` (absolute drop, in percentage points). | One project-wide convention to avoid subtle apples-to-oranges comparisons. |
| Label-fraction sweep | `{1, 5, 10, 50, 100}%`, with a per-class minimum-samples sanity check. | Standard SSL label-efficiency sweep. |
| Comparison rule | Paired test across matched seeds. An improvement counts as "real" only if its mean exceeds 1 × the std-dev band of the baseline AND the sign is positive. | Cheap and clean for a class project; avoids p-hacking. |
| Results directory | `results/<date>-<slice>-<short-description>/` containing `config.yaml`, `git_hash.txt`, `metrics.json`, and `notes.md`. | Per [`04-github-workflow.md`](04-github-workflow.md) section 7. |
| Code layout | Slice owners work under `src/slices/<owner-short-id>/`. Free to copy code between slices. No PR in one slice blocks another. | Strict slice independence. |
| Dataset storage | Raw data lives under `data/`, which is gitignored. Loaders download and cache. | Repo stays small; data does not leak via accidental commits. |

---

## 2. Per-component detail

### Encoder backbone

A small 1D CNN reads CSI shaped roughly `(time, subcarrier × antenna_pair)` and emits a single fixed-length vector. Three convolutional layers, global average pool, no residual connections. Around 50K parameters. A reference implementation will live at `src/scaffold/encoder.py` once the first slice's owner builds it; before that point, each slice writes its own and imitates the others.

We are not contributing on the architecture axis. The simplest backbone that produces non-trivial accuracy is the right backbone.

### SSL framework

SimCLR with the NT-Xent loss. Two augmented views per sample, contrastive objective pulls them together while pushing apart views of other samples in the batch. The augmentation pair is what each slice varies — that is the project's actual experimental variable.

Slices may compare against Barlow Twins, MAE, or others as a secondary experiment, but the headline number reported into the team's results table comes from SimCLR.

### Evaluation protocol

After SSL pre-training, the encoder is **frozen**. A single linear classifier is trained on top of it using labelled target-domain data. Accuracy on a held-out target-domain test split is the headline metric.

Linear probe is the SSL-literature default and it isolates "is the representation any good?" from "did fine-tuning rescue a mediocre representation?" If a slice wants to also report full fine-tuning numbers as a secondary metric, that is fine — note it in the writeup.

### Random seeds and reporting

Every headline number is the mean across at least 3 random seeds, reported with one standard deviation. Seeds are recorded in the per-experiment `notes.md`. Comparisons between configurations use paired tests across matched seeds: same seed → same data ordering → comparable runs.

The "improvement is real" rule is deliberately strict and simple: the candidate's mean must beat the baseline's `mean + 1 × std_dev` (and be positive in sign). This is more conservative than `p < 0.05` for small N and avoids p-hacking on a class-project budget.

### Cross-domain delta

Absolute drop in percentage points: `target_acc − source_acc`. We pick this over relative-drop or retention because absolute drop reads correctly across the wide accuracy range (50%–95%) we expect to see. Slice writeups always state the source-domain accuracy alongside the target-domain accuracy; the delta is a derived number, not a primary one.

### Code layout

Each slice owns a directory under `src/slices/<owner-short-id>/`. The owner's name is the short-id: `george/`, `chigozie/`, `collins/`, `ihunanya/`, `josiah/`, `victor/`.

Slice owners can freely copy code from other slices. There is no shared `src/scaffold/` library that everyone imports — that would create cross-slice dependencies. If a useful piece of scaffold-code emerges and stabilises after the slices land, the team's final closeout work can lift it into a shared location, but not before.

---

## 3. Opting out

If a slice needs a different backbone, SSL framework, or evaluation protocol, that is fine. The slice owner adds a one-paragraph **Convention deviations** section to their slice's results writeup (under `papers/team/<slice>.md`, or for Slice 1 in the KICS paper's discussion). The note must include:

1. Which default was deviated from.
2. Why the deviation was needed (compute, data shape, hypothesis-specific reason).
3. What was used instead.

Deviations are visible and reviewable without forcing a project-wide change.

---

## 4. AI-assisted work

Everything in [`06-using-ai-well.md`](06-using-ai-well.md) applies. The minimum standard there — every substantive claim either (a) backed by a paper you have read or (b) supported by a runnable experiment — is the standard for slice issue work too. AI can help write code, summarise papers, and propose experiments; AI cannot tell you whether your linear-probe accuracy is a bug or a result.
