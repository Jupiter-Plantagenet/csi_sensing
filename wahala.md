# Wahala — AFK plan vs reality

This file tracks places where `docs/slice-1-afk-plan.md` (written before the
Widar3.0 dataset and the local environment were examined) conflicts with
what's actually true on disk, in the data, or on the machine. Each conflict
is **recorded, not silently resolved** — George decides.

Each entry has:

- **Where**: section/line of the AFK plan (or other doc).
- **AFK plan says**: quoted from the plan.
- **Reality**: what's actually true.
- **What's shipped (if anything)**: the state already merged or under PR review.
- **Open question for George**: what needs resolving.

When a conflict is fully resolved (a side is picked and the doc/code agree),
the entry is removed and the resolution is folded back into the AFK plan or
the relevant doc.

---

## C1 — Branch base for tracer-bullet PRs

**Where.** `docs/slice-1-afk-plan.md` section 7, step 2.

**AFK plan says.** `git checkout main && git pull && git checkout -b george/T1.X-...`

**Reality.** The AFK plan itself, `docs/07-experiment-scaffold.md`, and
`docs/08-team-work-plan.md` live on the `george/team-work-plan` branch
(unmerged into `main`). The `papers/kics-george/` gitignore entry also
lives only on that branch. Branching off `main` literally means none of
those docs are on disk during the work.

**What's shipped.** T1.1 (PR #83) was branched off `main` per the literal
instruction. Practical impact for T1.1 is zero, but T1.7 will need either
the `papers/kics-george/` gitignore entry to be present or a manual
exclusion to keep `.tex/.pdf/.bib` out of the diff.

**Open question for George.** Branch the remaining tracer bullets off
`main` (literal plan), off `george/team-work-plan` (the doc-bearing
branch), or merge `george/team-work-plan` into `main` first?

---

## C2 — Encoder spec is internally inconsistent

**Where.** `docs/slice-1-afk-plan.md` section 7.1.

**AFK plan says.** "three Conv1d layers (64 → 128 → 256 channels) with
ReLU + BatchNorm, AdaptiveAvgPool1d(1), flatten to a 256-dim feature
vector. **Total ~50K parameters.**"

**Reality.** With the plan's own CSI input shape (S=30, A=3 → 90 input
channels) and kernel 3, those channel widths give ≈140K parameters, not
50K. To hit ~50K, the channels must be smaller (e.g. 32 → 64 → 128 ≈
40K), or the kernel/input shape must differ.

**What's shipped.** PR #83 uses 32 → 64 → 128 (~40K params) to honour
the project-wide "around 50K parameters" convention from
`docs/07-experiment-scaffold.md`. This was a silent judgment call;
flagging it now so it can be reverted if you'd rather honour the literal
channel widths.

**Open question for George.** Pick one and propagate to both the AFK plan
and `docs/07`:

- (a) keep 32 → 64 → 128, ~40K params (current);
- (b) widen back to 64 → 128 → 256 and accept ~140K params;
- (c) some other knob (smaller kernel, lower input dim via grouped conv,
      etc.).

---

## C3 — `results/` is gitignored but the AFK plan says it's project-tracked

**Where.** `docs/slice-1-afk-plan.md` section 1; `.gitignore` line 33.

**AFK plan says.** "All run logs under `results/<date>-george-*/` ...
each run directory contains `config.yaml`, `git_hash.txt`,
`metrics.json`, and `notes.md` — all tracked."

**Reality.** `.gitignore` excludes `results/` wholesale. New files under
`results/` will be silently ignored by `git add` and won't appear in
diffs.

**What's shipped.** Nothing yet — T1.1 doesn't generate results. T1.3
hits this for the first time.

**Open question for George.** Pick before T1.3:

- (a) replace `results/` with negation patterns that allow the four
      tracked text files;
- (b) drop `results/` from `.gitignore` entirely and rely on convention
      to keep checkpoints (`*.pth`, `*.pt`, `*.ckpt` already gitignored)
      out;
- (c) `git add -f` the four files per run (works but easy to forget).

---

## C4 — CUDA-detection branch was wrong; the GPU is real

**Where.** `docs/slice-1-afk-plan.md` sections 4.3 and 5.

**AFK plan says.** "If `cuda: False`, you are on CPU. Read section 5 —
your compute strategy changes."

**Reality.** `nvidia-smi` reports an NVIDIA GeForce RTX 5060 (8 GB,
driver 591.86, CUDA 13.1 runtime). `torch.cuda.is_available()` returned
`False` because the installed wheel is `torch==2.10.0+cpu`, not because
the hardware is missing. The plan's check is sufficient for "is the
wheel right?" but not for "is there a GPU?".

**What's shipped.** None. T1.1 ran on the CPU wheel — fine for stub
data. Without a fix, T1.3 onward would have defaulted to "reduced
experimental scope" (4-subject Widar3.0 subset on CPU), which would
have been the wrong call.

**Open question for George.** Approve installing a CUDA-enabled torch
wheel (~3 GB download)? RTX 5060 is Blackwell (sm_120), so the
recommended index is the latest CUDA 12.8 wheel:

```
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

Once installed, AFK plan section 5.1's budget applies: ~30 min/run, ~3
hours for the full 6-run multi-seed comparison.

---

## C5 — Widar3.0 dataset specifics not yet verified

**Where.** `docs/slice-1-afk-plan.md` section 6.1.

**AFK plan says.** CSI shape `(T, S, A)` with `S=30, A=antenna_pair`;
cross-subject test split `[1, 2, 3, 4]`; six activity classes;
`train_subjects` parameter; etc.

**Reality.** Not yet verified — I haven't downloaded Widar3.0 or
inspected a sample. George flagged that the AFK plan was written before
the dataset was examined, so any of these specifics may need correction
once T1.2 starts. Likely affected fields: `NUM_CLASSES = 6` (currently
hard-coded in `src/slices/george/data.py`), the exact `(T, S, A)`
tensor shape, sample length variability, and whether the dataset
publishes a canonical cross-subject split.

**What's shipped.** None — flagging proactively. T1.2 will add concrete
corrections to this entry rather than silently picking.

**Open question for George.** Which Widar3.0 source/format should I
use?

- (a) the Tsinghua official release (`.dat` files; needs the published
      MATLAB / Python loader);
- (b) a Hugging Face mirror, if one exists;
- (c) a pre-processed `.npy` archive from a third-party loader;
- (d) something already cached locally on this machine that I should
      point at.

Also: do you already have a preferred cross-subject split convention
(e.g. from a published paper) you want me to use, or should I default
to `[1, 2, 3, 4]` as test subjects per the AFK plan?

---

## C6 — `papers/kics-george/` gitignore entry is missing on `main`

**Where.** `docs/slice-1-afk-plan.md` section 1; `.gitignore`.

**AFK plan says.** "Everything under `papers/kics-george/` ...
gitignored by the project's existing `.gitignore`."

**Reality.** On `main` (current base of T1.1 and this branch), the
`.gitignore` does **not** have a `papers/kics-george/` entry. The entry
only exists on the `george/team-work-plan` branch.

**What's shipped.** None — won't matter until T1.7 (paper draft) but
worth flagging now since it's the same root cause as C1.

**Open question for George.** Add the `papers/kics-george/` line to
`.gitignore` in T1.7's PR, merge `george/team-work-plan` into `main`
first so the entry lands by other means, or accept the resolution from
C1?

---

## Add new entries below as conflicts surface.
