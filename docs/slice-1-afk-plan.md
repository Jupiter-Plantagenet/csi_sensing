# Slice 1 AFK Plan — George's KICS Submission

This document is for Claude running in a fresh session, working autonomously while George is away from keyboard, with the goal of executing Slice 1 of the project and producing a KICS-submittable 2-page paper.

If you (Claude) are reading this in a new session: the plan is self-contained. Read sections 0–4 first, then execute T1.1–T1.8 in order. The repository's other docs are referenced when needed; you do not need to read them all upfront.

If George (the user) is reviewing this document: the plan is calibrated for "Claude executes autonomously, only pauses at the explicit STOP points." Sanity-check the STOP points (section 9), the compute strategy (section 4), and the negative-result protocol (section 11). Edits welcome.

---

## 0. Who you are and what you are doing

**You are Claude, in a fresh session, working on the repository at [`https://github.com/Jupiter-Plantagenet/csi_sensing`](https://github.com/Jupiter-Plantagenet/csi_sensing).** The user George (`Jupiter-Plantagenet` on GitHub) has stepped away. You have authenticated `gh` CLI access as George and full write access to the repo.

**Your goal**: complete Slice 1 of the project's six-slice plan. Slice 1 produces a 2-page KICS conference paper on Doppler-aware time warping for cross-subject WiFi CSI sensing. The slice is broken into eight tracer-bullet issues (#34 through #41). You execute them in order, opening one PR per tracer bullet, committing progress as you go.

**You stop at the explicit STOP points in section 9.** The most important is: **never click "Merge" on a PR**. George reviews and merges. You also **do not** upload the final paper to the KICS submission portal — that is George's manual step. You produce the submission-ready PDF and stage it; George submits.

---

## 1. What is committed and what is not

Read this carefully — it shapes how every PR you open is structured.

**Committed (project-tracked):**

- All code under `src/slices/george/`.
- All unit tests under `tests/slices/george/`.
- All run logs under `results/<date>-george-*/`. Per [`docs/07-experiment-scaffold.md`](07-experiment-scaffold.md), each run directory contains `config.yaml`, `git_hash.txt`, `metrics.json`, and `notes.md` — all tracked.
- Figure-generation scripts (e.g. `src/slices/george/figures.py` if you produce figures programmatically).
- This AFK plan and any updates to it.
- A short status pointer file at `papers/team/george-kics-status.md` that records where the locally-built paper stands (created and updated as you progress through T1.7 and T1.8).

**Not committed (local-only, gitignored):**

- Everything under `papers/kics-george/`. This includes `main.tex`, `main.pdf`, `refs.bib`, `fig1-pipeline.pdf`, `SUBMISSION.md`, and any other paper-side artifact.

The reason: the slice's *engineering work* is project-tracked so teammates can review and reuse it. The *paper itself* is George's personal academic output, kept out of the project's git history. The paper still gets produced — it just lives locally on whichever machine the AFK session ran on.

**Implication for your PRs.** Each tracer-bullet PR commits only project-tracked artifacts. T1.7 and T1.8 are the unusual case: the paper itself (the .tex/.pdf/.bib) does not appear in the diff. Those PRs commit the figure script, the results, and an updated `papers/team/george-kics-status.md`. The paper's existence is referenced in the PR description with the local path; the actual files do not get pushed.

**One-machine implication.** Because paper artifacts live locally and are not in git, the AFK session must run on a machine where the paper files persist. If you need to resume on a different machine, regenerate the paper from the project-tracked figure scripts and results, and re-run the LaTeX compilation. There is no automatic sync — that is George's problem to solve (e.g., Drive sync) if he wants cross-machine continuity.

---

## 2. Required reading order

Read these in order. Stop after each, do not switch tabs:

1. This document, all of it.
2. [`docs/08-team-work-plan.md`](08-team-work-plan.md), specifically section 4 (Slice 1) — the slice-specific roadmap and the KICS paper outline.
3. [`docs/07-experiment-scaffold.md`](07-experiment-scaffold.md) — encoder, SSL, eval defaults you must follow.
4. [`docs/06-using-ai-well.md`](06-using-ai-well.md) — when *you* run AI-flavoured operations on yourself (web search, paper summarisation), follow this.
5. [`docs/04-github-workflow.md`](04-github-workflow.md) — branch, commit, PR conventions.
6. The eight tracer-bullet issues #34–#41 on GitHub — read each issue body in full before starting work on it.

You do not need to read docs 01–03 unless you hit a conceptual question about CSI, SSL, or the project's research thesis. They are background, not instructions.

---

## 3. The deliverable in one paragraph

A 2-page IEEE-conference-format LaTeX paper, compiled to PDF, with a comparison table showing that **Doppler-aware time warping** (random time-axis stretch by a factor in `[0.7, 1.4]`) used as a SimCLR augmentation pair-half during self-supervised pre-training of a small CNN encoder on Widar3.0 cross-subject **improves linear-probe accuracy** vs a generic-augmentation baseline (Gaussian noise + random subcarrier mask) across three random seeds. The paper follows the structure of George's prior KICS Winter 2026 paper (`Towards Quantization-Native Zero-Knowledge Verification for INT4 Large Language Model Inference`). Output artifacts live under local-only `papers/kics-george/`. The submission upload itself is George's manual step. The project-tracked artifacts that surface this paper's existence are: the slice code, the run logs, the figure scripts, and a status pointer at `papers/team/george-kics-status.md`.

If the experimental result is **negative** (Doppler does not beat the baseline within the convention rule in [`docs/07-experiment-scaffold.md`](07-experiment-scaffold.md)), the paper still gets written — see section 11.

---

## 4. Environment setup (one-time, in order)

Run each command, verify success, do not skip steps. If a step fails, do not work around it — comment on issue #34 with the failure and STOP.

### 4.1 Verify the basics

```bash
cd c:/Users/user/CascadeProjects/csi_sensing
git status                           # clean working tree (or only the .pptx untracked)
git fetch origin
gh auth status                       # logged in as Jupiter-Plantagenet
python --version                     # 3.10 or newer
```

### 4.2 Create a Python virtual environment

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Or (Git Bash):
source .venv/Scripts/activate

pip install --upgrade pip wheel
pip install torch numpy scipy matplotlib pyyaml tqdm scikit-learn
pip install pytest                   # for sanity tests
# Note: GPU torch wheel install if a CUDA GPU is detected; otherwise CPU is fine
```

If `pip install torch` is slow, use the official PyTorch index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu        # CPU-only
# or for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 Verify CUDA availability

```python
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
```

If `cuda: False`, you are on CPU. Read section 5 — your compute strategy changes.

### 4.4 LaTeX

You need LaTeX to produce the final PDF. Three options, in preference order:

1. **Local TeX Live / MiKTeX** — preferred. Run `pdflatex --version`. If it works, you are set. If you hit a missing-package error during compile, install the package via `tlmgr install <pkg>` (TeX Live) or the MiKTeX console.
2. **Overleaf** — if no local LaTeX. Create a project, upload the `.tex` source, compile online, download PDF. Account creation may be needed.
3. **`tectonic`** — `pip install tectonic` or `cargo install tectonic`. Self-contained TeX engine; downloads packages on demand. Good fallback when MiKTeX is uncooperative.

You only need LaTeX for T1.7 and T1.8. Do not block earlier tracer bullets on this.

---

## 5. Compute strategy

Pre-training even a small SimCLR model on Widar3.0 takes meaningful compute. Plan accordingly.

### 5.1 If a CUDA GPU is available

You can do everything locally. Budget ~30 minutes per pre-training run (50 epochs, batch size 256, ~50K-param encoder). Three seeds × two conditions = 6 runs ≈ 3 hours.

### 5.2 If you are on CPU only

Local CPU is too slow for full pre-training. Options:

1. **Google Colab (free)** — fastest path. Upload the `src/slices/george/` package as a zip, mount Drive for caching, run pre-training there, download the trained checkpoint. Free tier gives a T4 GPU for ~12-hour sessions. **Do not push large checkpoints to git** — they go to Drive or are kept ephemeral.
2. **Kaggle Notebooks (free)** — similar idea. Persistent Kaggle datasets can host Widar3.0.
3. **Reduced experimental scope** — use a 4-subject subset of Widar3.0 (instead of all 17) for the tracer bullets and final results. The paper's claim becomes "preliminary on a 4-subject subset" with explicit framing in the abstract. This is acceptable for KICS and aligns with the "Towards" framing of George's prior paper.

**Decision rule:** if `torch.cuda.is_available()` is False, default to option 3 (reduced scope on CPU) and document the choice in a comment on issue #36 (T1.3, where real experiments start). Do **not** spend money on cloud GPU without George's explicit approval — see section 9.

### 5.3 Stub vs real for tracer bullets

T1.1 (scaffold) **must** run on stub data — 10 random tensors of CSI-shaped values. No real data needed. CPU is fine.

T1.2 onward uses real data; this is where compute matters.

---

## 6. Dataset access

### 6.1 Widar3.0 — primary dataset

Widar3.0 is hosted by the Tsinghua / SyNet group. The canonical access pattern in the WiFi-sensing literature:

1. Visit the official Widar3.0 page (search: "Widar 3.0 dataset Tsinghua").
2. Some access points require registration; some are public mirrors. Use Google to find a working public mirror if the official one is gated.
3. Common forms: `.dat` files (raw CSI from Intel 5300) or pre-processed `.npy` arrays from a third-party loader.

**Cache location:** `data/widar3/` (gitignored by the project's existing `.gitignore`). Do not commit raw data.

**Schema:** the loader at `src/slices/george/data.py` (you will write it in T1.2) needs to expose:

```python
class Widar3CrossSubject:
    def __init__(self, root: str, train: bool = True, train_subjects: list[int] = ...): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> tuple[torch.Tensor, int]:
        # returns (csi_tensor, label)
        # csi_tensor shape: (T, S, A) where T=time, S=subcarrier, A=antenna_pair
        # label: int 0..K-1 for K activity classes
```

The cross-subject split convention: pick `[1, 2, 3, 4]` as test subjects, the rest as train. Hard-code this in T1.2 unless a published cross-subject split convention is found.

### 6.2 If Widar3.0 is unavailable

Fall back, in order:

1. **A Widar3.0 mirror on Hugging Face Datasets** — search "widar" or "widar3" on huggingface.co/datasets. If found, the loader becomes a thin wrapper over `datasets.load_dataset`.
2. **UT-HAR** as a substitute. Smaller, single-environment dataset. Adjust the paper's framing to "evaluated on UT-HAR within-domain with a held-out user split" — note this requires updating the abstract and conclusions.
3. **Synthetic CSI** as a last resort. Define a synthetic generator that produces CSI-like signals with a known Doppler structure, and run the entire pipeline on synthetic data. The paper becomes a methods-paper rather than a benchmark paper. **Document this clearly in the title and abstract** — do not pretend synthetic results are real.

If you fall back to (2) or (3), comment on issue #35 explaining the fallback and **STOP** — wait for George to confirm the change in scope.

### 6.3 CSI-Bench

Not needed for Slice 1. Ignore.

---

## 7. The eight tracer bullets

For each tracer bullet, the steps are:

1. Read the issue body on GitHub: `gh issue view <NUM>`
2. Create a branch: `git checkout main && git pull && git checkout -b george/T1.X-short-description`
3. Implement, committing in small, reviewable chunks
4. Run the self-review checklist (section 9.4)
5. Push: `git push -u origin <branch>`
6. Open PR: `gh pr create --fill --assignee Jupiter-Plantagenet`
7. Comment on the corresponding issue with the PR link
8. **STOP and wait for George to merge.** Do not proceed to the next tracer bullet until the PR is merged. (See section 9 — this is the key STOP point.)

**Important:** The PR being open is the trigger for George to review. Do not open multiple PRs at once and stack work on top of unmerged PRs. One tracer bullet, one PR, one review-and-merge cycle.

### 7.1 T1.1 — Scaffold + SimCLR end-to-end with stub data (issue #34)

**Goal.** A runnable Python package under `src/slices/george/` that defines a tiny CNN encoder, a SimCLR loss, a linear-probe evaluator, and a `run.py` that strings them together on 10 random stub samples and prints a number. Number need not be meaningful — pipeline must run.

**Default approach.**

- `src/slices/george/__init__.py` — empty.
- `src/slices/george/encoder.py` — class `TinyCNN(nn.Module)`, three Conv1d layers (64 → 128 → 256 channels) with ReLU + BatchNorm, AdaptiveAvgPool1d(1), flatten to a 256-dim feature vector. Total ~50K parameters. The CSI tensor is reshaped to `(B, S*A, T)` before the conv stack — subcarrier × antenna_pair becomes the channel dim, time becomes the conv axis.
- `src/slices/george/ssl.py` — `class SimCLR` wrapping the encoder + a 2-layer MLP projection head. NT-Xent loss with temperature 0.5.
- `src/slices/george/eval.py` — `linear_probe(encoder, train_loader, test_loader)` freezes the encoder, fits a linear classifier on its outputs, returns test accuracy.
- `src/slices/george/data.py` — `class StubCSI` returning 10 random tensors shaped `(100, 30, 3)` (T, S, A) with random labels in `{0, 1, 2, 3, 4, 5}`.
- `src/slices/george/run.py` — entrypoint: build encoder, build SimCLR wrapper, pre-train for 2 epochs on `StubCSI`, run `linear_probe`, print final accuracy.
- `tests/slices/george/test_smoke.py` — `pytest` test that imports `run` and runs `main()` without crashing.
- `src/slices/george/README.md` — one-paragraph overview, how to run.

**What done looks like.**

- `python -m src.slices.george.run` prints a number (probably ~16% — chance for 6 classes).
- `pytest tests/slices/george/test_smoke.py` passes.
- CI green on the PR.

**Notes.**

- Do not use the project's `src/scaffold/` directory yet — it does not exist. Slice 1 may pioneer the encoder; future slices can copy.
- Use `torch.compile` only if it works out of the box; otherwise skip.
- Keep all code Python 3.10-compatible.

### 7.2 T1.2 — Real Widar3.0 cross-subject loader (issue #35)

**Goal.** Replace `StubCSI` with a real Widar3.0 cross-subject loader. The same `run.py` from T1.1 should now produce above-chance accuracy.

**Default approach.**

1. Find Widar3.0 (see section 6).
2. Implement `Widar3CrossSubject(root, train, train_subjects=...)` per section 6.1 schema.
3. Cache parsed data to `data/widar3/cache/<split>.pt` to avoid re-parsing on every run.
4. Update `run.py` to use the real loader. Bump epochs to 50, batch size to 256.
5. Sanity-check: print the first sample's shape and label.

**What done looks like.**

- Linear-probe accuracy on cross-subject test ≥ 17% (chance) and ideally ≥ 25% (showing the encoder learned something even with no augmentation).
- Loader handles the case where data is already cached (do not re-download or re-parse).
- A small README at `src/slices/george/data.py` docstring or `data/widar3/README.md` documents how to obtain Widar3.0.

**Notes.**

- If Widar3.0 download path returns 404 or is gated and you cannot resolve it, see section 6.2 fallback chain. **Comment on the issue and STOP** before falling back to synthetic data.

### 7.3 T1.3 — Generic-augmentation baseline (issue #36)

**Goal.** Run SimCLR with two generic augmentations (Gaussian noise + random subcarrier masking) as the augmentation pair, single seed, and record the cross-subject linear-probe accuracy. This is the baseline George's Doppler aug needs to beat.

**Default approach.**

- `src/slices/george/augmentations.py`:
  - `gaussian_noise(x, sigma=0.05)` — additive Gaussian noise.
  - `random_subcarrier_mask(x, p=0.15)` — zero out a random fraction of subcarriers.
- In `ssl.py`, the SimCLR view-pair function applies one randomly to view 1 and the other to view 2.
- Single seed (`SEED=42` per [`docs/04-github-workflow.md`](04-github-workflow.md)).
- Log result to `results/<YYYY-MM-DD>-george-baseline-singleseed/` with `config.yaml`, `git_hash.txt`, `metrics.json`, `notes.md`.

**What done looks like.**

- A reproducible cross-subject linear-probe accuracy number, printed and logged.
- The `results/` directory contains all four required files per [`docs/07-experiment-scaffold.md`](07-experiment-scaffold.md).

### 7.4 T1.4 — Doppler-aware time warping augmentation (issue #37)

**Goal.** Implement Doppler-aware time warping. Use it as one half of the SimCLR view pair (the other half stays as the generic augmentation, OR also Doppler — see notes). Record cross-subject linear-probe accuracy, single seed.

**Default implementation.**

```python
def doppler_warp(x: torch.Tensor, factor: float = None) -> torch.Tensor:
    """Stretch CSI's time axis by a random factor in [0.7, 1.4].

    x: (T, S, A)
    factor: if None, sampled uniformly from [0.7, 1.4]
    returns: (T, S, A) — same length as input, via interpolation
    """
    if factor is None:
        factor = float(torch.empty(1).uniform_(0.7, 1.4))
    T, S, A = x.shape
    t_orig = torch.linspace(0, 1, T)
    t_warped = torch.linspace(0, 1, int(T * factor))
    # resample x along time axis
    x_r = torch.nn.functional.interpolate(
        x.permute(1, 2, 0).unsqueeze(0),  # (1, S, A, T)
        size=int(T * factor),
        mode='linear',
        align_corners=False,
    ).squeeze(0).permute(2, 0, 1)
    # crop or pad back to T
    if x_r.shape[0] >= T:
        return x_r[:T]
    else:
        pad = torch.zeros(T - x_r.shape[0], S, A)
        return torch.cat([x_r, pad], dim=0)
```

**View-pair strategy.** Two reasonable choices, document the one you pick:

1. **Asymmetric:** view 1 = generic baseline aug, view 2 = Doppler. Compares Doppler against generic in a direct contrast.
2. **Symmetric:** both views = Doppler with independently sampled factors. Tests Doppler in isolation.

The default for the comparison table is **symmetric Doppler vs symmetric generic** — that is what gives a clean apples-to-apples comparison.

**What done looks like.**

- Doppler aug runs without crashing.
- Cross-subject linear-probe accuracy logged to `results/<date>-george-doppler-singleseed/`.

### 7.5 T1.5 — Sanity test (issue #38)

**Goal.** Unit test confirming `doppler_warp(x, factor=2.0)` shifts a synthetic CSI's dominant frequency by approximately 2×.

**Default approach.**

```python
# tests/slices/george/test_doppler.py
import torch
from src.slices.george.augmentations import doppler_warp

def test_doppler_doubles_frequency():
    T, S, A = 256, 1, 1
    f0 = 8.0  # cycles over the window
    t = torch.linspace(0, 1, T)
    x = torch.sin(2 * torch.pi * f0 * t).reshape(T, S, A)

    y = doppler_warp(x, factor=2.0)

    fft_x = torch.fft.rfft(x[:, 0, 0]).abs()
    fft_y = torch.fft.rfft(y[:, 0, 0]).abs()
    peak_x = int(fft_x.argmax())
    peak_y = int(fft_y.argmax())

    # factor 2.0 stretches time → halves frequency in the warped sample's frame
    # but the warp is applied AND re-cropped to T, so the dominant frequency
    # of the cropped result corresponds to ~f0/2 in the new frame.
    # Verify: peak_y is approximately peak_x // 2 (within tolerance).
    assert abs(peak_y - peak_x // 2) <= 2, f"expected peak near {peak_x//2}, got {peak_y}"
```

**Notes.** The exact direction of the frequency shift depends on whether the warp is *resample-then-crop* or *resample-and-rescale*. The test expectation must match the actual implementation. Adjust the assertion if the test fails — but only after confirming the implementation does what it claims.

### 7.6 T1.6 — Multi-seed comparison (issue #39)

**Goal.** Run T1.3 and T1.4 with three random seeds each. Compute mean ± std. Apply the convention rule from [`docs/07-experiment-scaffold.md`](07-experiment-scaffold.md) section 1: improvement is "real" if `mean(ours) > mean(baseline) + std(baseline)`.

**Default approach.**

- Three seeds: `[42, 1337, 2024]`.
- Re-run baseline (T1.3) and Doppler (T1.4) under each seed.
- Aggregate results into a markdown table at `results/<date>-george-3seed-summary/notes.md` (project-tracked) AND mirror into local-only `papers/kics-george/results.md` (for paper consumption).

**What done looks like.**

- Six total runs (or twelve, if you compute both linear-probe-from-scratch and full fine-tune; default is just linear probe).
- A table like:

| Configuration | Cross-subject acc (mean ± std) |
|---|---|
| No augmentation | XX.X ± Y.Y |
| Gaussian noise + random mask (baseline) | XX.X ± Y.Y |
| **Doppler-aware time warping (ours)** | **XX.X ± Y.Y** |

- A line of prose stating whether the improvement passes the convention rule.

### 7.7 T1.7 — KICS paper draft (issue #40)

**Goal.** Produce `papers/kics-george/main.tex`, `papers/kics-george/refs.bib`, and a compiled `papers/kics-george/main.pdf` that follows the structure in [`docs/08-team-work-plan.md`](08-team-work-plan.md) section 4 KICS outline.

**Project-tracked artifacts (these go in the PR diff):**

- `src/slices/george/figures.py` — script that produces `fig1-pipeline.pdf` (or `.png`). Same figure output, but the script is what gets reviewed; the rendered figure lives locally only.
- `papers/team/george-kics-status.md` — short status pointer file. New on this PR. Three sections: "draft state" (e.g. "draft compiled, 2 pages, references unverified"), "local paper path" (e.g. "`papers/kics-george/main.pdf` on George's machine"), "notes for reviewer" (e.g. flags about placeholder figures, missing citations).
- Any updates to `results/` if extra runs were needed for the figure.

**Local-only artifacts (these do NOT go in the PR):**

- `papers/kics-george/main.tex`, `papers/kics-george/refs.bib`, `papers/kics-george/main.pdf`, `papers/kics-george/fig1-pipeline.pdf`. All under the gitignored directory.

See section 12 for IEEEtran setup, author list, acknowledgment, and reference template — all the boilerplate that is hard to reconstruct without the prior paper in front of you.

**What done looks like.**

- `main.tex` compiles cleanly with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.
- The PDF is exactly 2 pages.
- The compiled PDF is at `papers/kics-george/main.pdf` (local).
- One architecture figure (Fig. 1) is rendered at `papers/kics-george/fig1-pipeline.pdf` (local) by the project-tracked `src/slices/george/figures.py`.
- `papers/team/george-kics-status.md` exists and accurately describes the draft state.

### 7.8 T1.8 — KICS paper polish and submission staging (issue #41)

**Goal.** A submission-ready PDF, with all references verified, format checked, and a checklist for George to do the actual portal upload.

**Steps.**

1. Verify each reference: open the cited paper, confirm it says what the citation claims it says. Per [`docs/06-using-ai-well.md`](06-using-ai-well.md), do not skip this. Fake citations are how papers get rejected.
2. Verify the 2-page hard limit on the rendered PDF.
3. Verify the acknowledgment block matches the prior paper exactly where text is reused.
4. Write `papers/kics-george/SUBMISSION.md` (local-only) — a one-page checklist for George with: target KICS edition, submission portal URL, deadline, acknowledgment of where the PDF lives, any known caveats about the experimental setup, and a "human steps remaining" list.
5. Update `papers/team/george-kics-status.md` (project-tracked) to "submission staged; awaiting human upload" with the local PDF path and a one-line summary of the result.

**Project-tracked PR diff:** updated `papers/team/george-kics-status.md`, plus any final tweaks to figure scripts or run logs.

**STOP after T1.8 PR is opened.** Do not attempt the actual submission. George does that.

---

## 8. Branch and commit conventions

- Branch names: `george/T1.X-short-description` (e.g., `george/T1.1-scaffold`, `george/T1.4-doppler-aug`).
- Commits: imperative voice, ≤72-char subject. Body explains *why* if non-obvious. Examples:
  - `T1.1: Add tiny CNN encoder and SimCLR loss skeleton`
  - `T1.4: Implement Doppler-aware time warping augmentation`
  - `T1.6: Run multi-seed comparison; Doppler beats baseline by 2.3%`
- One PR per tracer bullet. Link the issue with `Closes #34` (or whichever number) in the PR body.
- PR title: `T1.X — <short description>`. Body: 3–6 sentences on what the PR does, what was tested, and any open questions. For T1.7 / T1.8, include the local path to the compiled paper PDF in the PR body.

---

## 9. STOP points and self-review

### 9.1 STOP points (never proceed past these without George's input)

- **Each opened PR is a STOP point.** Wait for George to merge before opening the next tracer bullet's PR.
- **Widar3.0 unavailable.** Comment on issue and STOP. Section 6.2 fallback chain requires George's explicit OK.
- **Negative experimental result** (Doppler does not beat baseline by the convention rule). Comment on issue #39, write up the result honestly, and STOP before T1.7. The paper still gets written but the framing changes — see section 11.
- **Compute budget overrun.** If a single tracer bullet's experiments would take >12 hours of wall-clock or any cloud cost, STOP and ask.
- **LaTeX setup blocker.** If you cannot get any of the three LaTeX paths in section 4.4 working, STOP. Do not invent a workflow.
- **The KICS submission portal.** You produce the PDF and the SUBMISSION.md checklist. George uploads. Period.
- **Modifying any other slice's code.** Slices 2–6 are owned by other teammates. Do not touch their `src/slices/<owner>/` directories.
- **Adding new collaborators or changing repo settings.** Don't.

When you stop, the protocol is:

1. Push whatever progress exists to the current branch.
2. Comment on the relevant issue with: what you did, what you saw, why you stopped, what input you need from George.
3. Stop generating code. Wait.

### 9.2 What you may do without asking

- Read any file in the repo.
- Open issues for unexpected sub-questions you discover. Use the `Task` template. Tag `slice:george-doppler`.
- Run experiments locally on whatever hardware is available, using free tiers of cloud services if needed (Colab free, Kaggle free).
- Comment on existing issues with progress notes.
- Open PRs against tracer-bullet branches.
- Modify (gitignored) `papers/kics-george/` freely.
- Modify `src/slices/george/` freely.
- Add unit tests under `tests/slices/george/`.
- Update this AFK plan if you discover something the next AFK session would benefit from knowing — but do so in a separate PR.

### 9.3 Self-review checklist (run before every PR)

Run all of these. Do not open the PR until they pass.

1. **Code runs.** `python -m <entrypoint>` works without errors.
2. **Tests pass.** `pytest tests/slices/george/` is green.
3. **Lint passes.** `black --check src/slices/george/ tests/slices/george/` and `ruff check src/slices/george/ tests/slices/george/`. Fix anything they flag.
4. **No accidental commits.** `git diff main..HEAD` only shows the intended changes. No `__pycache__`, no `.venv`, no `data/widar3/`, **no `papers/kics-george/` files** (those are gitignored — confirm by running `git status` and ensuring nothing under `papers/kics-george/` shows up).
5. **Results are logged.** If this PR runs experiments, the `results/` directory has the four required files (`config.yaml`, `git_hash.txt`, `metrics.json`, `notes.md`). The `notes.md` is written by you, in plain prose, describing what you expected and what you saw.
6. **Issue is referenced.** PR body says `Closes #N`.
7. **For T1.7 and T1.8 only:** the PR diff contains only project-tracked artifacts (figure script, status pointer, results). The `.tex/.pdf/.bib` files are in your local working directory but not staged.

---

## 10. Negative-result protocol

If after T1.6, the Doppler-aware augmentation does not beat the baseline within the convention rule:

1. Do **not** retroactively change the experimental design (different seeds, different aug parameters, different encoder) to "find" a positive result. That is data dredging.
2. Comment on issue #39 with the full result table and the conclusion.
3. STOP. Wait for George to decide whether to:
   - **Proceed with the negative result.** The paper becomes "Towards Physics-Informed Augmentation: A Negative Result on Cross-Subject WiFi CSI Sensing" or similar. KICS accepts negative results, especially with "Towards" framing. The paper structure in section 12 is largely unchanged — only the conclusion and abstract framing shift.
   - **Iterate one parameter** before re-evaluating (e.g., warp range `[0.5, 1.7]` instead of `[0.7, 1.4]`). At most one iteration; if that fails, the result is genuinely negative.
   - **Pivot** the paper to a different angle (e.g., comparing several augmentations all of which underperform). This requires significant scope change and is George's call.

A negative result is publishable, useful, and honest. Treat it as such.

---

## 11. KICS paper specifics

### 11.1 IEEEtran template

Get `IEEEtran.cls` from CTAN: [https://www.ctan.org/pkg/ieeetran](https://www.ctan.org/pkg/ieeetran). MiKTeX and TeX Live ship it. Header for `main.tex` (lives at the local-only `papers/kics-george/main.tex`):

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{Towards Physics-Informed Augmentation for Cross-Subject WiFi CSI Sensing: \\
Doppler-Aware Time Warping in Self-Supervised Pre-Training}

\author{
\IEEEauthorblockN{George Chidera Akor\IEEEauthorrefmark{1},
                  Love Allen Chijioke Ahakonye\IEEEauthorrefmark{2},
                  Jae Min Lee\IEEEauthorrefmark{1},
                  Dong-Seong Kim\IEEEauthorrefmark{1}\IEEEauthorrefmark{3}}
\IEEEauthorblockA{\IEEEauthorrefmark{1}IT-Convergence Engineering, Kumoh National Institute of Technology, Gumi, South Korea}
\IEEEauthorblockA{\IEEEauthorrefmark{2}ICT Convergence Research Center, Kumoh National Institute of Technology, Gumi, South Korea}
\IEEEauthorblockA{\IEEEauthorrefmark{3}NSLab Co. Ltd., Gumi, South Korea}
\IEEEauthorblockA{\{georgeakor, loveahakonye, ljmpaul, dskim\}@kumoh.ac.kr}
}

\maketitle

\begin{abstract}
% bottleneck → observation → present → numbers (~150 words)
\end{abstract}

\begin{IEEEkeywords}
Channel state information, self-supervised learning, Doppler shift, data augmentation, domain generalization, WiFi sensing.
\end{IEEEkeywords}

\section{Introduction}
% ~half page
\section{Methodology}
\subsection{Doppler scaling and the time-warp operation}
\subsection{SSL pipeline integration}
\section{Results and Discussion}
\section{Conclusion}

\section*{Acknowledgment}
% paste section 11.3 verbatim

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
```

### 11.2 Author list (verbatim — do not modify without explicit approval)

```
George Chidera Akor (Kumoh, IT-Convergence Engineering)
Love Allen Chijioke Ahakonye (Kumoh, ICT Convergence Research Center)
Jae Min Lee (Kumoh, IT-Convergence Engineering)
Dong-Seong Kim (Kumoh, IT-Convergence Engineering; NSLab Co. Ltd.)
```

Emails: `georgeakor`, `loveahakonye`, `ljmpaul`, `dskim` all `@kumoh.ac.kr`.

### 11.3 Acknowledgment block (paste verbatim in `\section*{Acknowledgment}`)

```
This work was partly supported by the Innovative Human Resource Development for Local Intellectualization program through the IITP grant funded by the Korea government (MSIT) (IITP-2025-RS-2020-II201612, 25\%), by the Priority Research Centers Program through the NRF funded by the MEST (2018R1A6A1A03024003, 25\%), and by the MSIT, Korea, under the ITRC support program (IITP-2025-RS-2024-00438430, 25\%) and by the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (RS-2025-25431637, 25\%).
```

(Note the `\%` escapes for percent signs in LaTeX.)

### 11.4 Reference list (start from these, verify each)

In `papers/kics-george/refs.bib` (local-only). Confirm each by opening the actual paper. Use Google Scholar or arXiv to find the canonical bib entry.

```bibtex
@article{yang2023autofi,
  title={AutoFi: Toward Automatic WiFi Human Sensing via Geometric Self-Supervised Learning},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Lu, Chris Xiaoxuan and Wang, Dazhuo and Sun, Sumei and Xie, Lihua},
  journal={IEEE Internet of Things Journal},
  year={2023},
}

@article{barahimi2024capc,
  title={Context-Aware Predictive Coding: A Representation Learning Framework for WiFi Sensing},
  author={Barahimi, B. and others},
  journal={IEEE Open Journal of the Communications Society},
  year={2024},
}

@inproceedings{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020},
}

@inproceedings{zheng2019widar3,
  title={Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi},
  author={Zheng, Yue and Zhang, Yi and Qian, Kun and Zhang, Guidong and Liu, Yunhao and Wu, Chenshu and Yang, Zheng},
  booktitle={MobiSys},
  year={2019},
}

@article{xu2025ssl,
  title={Evaluating Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition},
  author={Xu, K. and others},
  journal={ACM Transactions on Sensor Networks},
  year={2025},
}

@inproceedings{akor2025purechain,
  title={Purechain-based Zero-Knowledge Proofs for Verifiable Machine Learning in Industrial IoT},
  author={Akor, George Chidera and Ahakonye, Love Allen Chijioke and Lee, Jae Min and Kim, Dong-Seong},
  booktitle={KICS Fall Conference},
  year={2025},
}

@inproceedings{akor2026ezkl,
  title={Benchmarking CNN Components in EZKL: A Layer-Level Analysis for EVM-Compatible Deployment},
  author={Akor, George Chidera and Ahakonye, Love Allen Chijioke and Lee, Jae Min and Kim, Dong-Seong},
  booktitle={ICAIIC},
  year={2026},
}
```

Add 1–2 more references if space allows: a domain-generalization survey, or CIG-MAE (cross-modal MAE for WiFi sensing) for completeness.

### 11.5 The architecture figure (Fig. 1)

Match the visual style of George's prior paper Fig. 1. Components, left to right:

1. A box labelled **Widar3.0 CSI sample** (input).
2. Two parallel arrows to two boxes labelled **View 1: Doppler warp (factor f₁)** and **View 2: Doppler warp (factor f₂)**.
3. Both feed into a box labelled **Tiny CNN encoder (f_θ)**.
4. Encoder outputs feed into a box labelled **NT-Xent loss** (SimCLR).
5. After pre-training, a separate downstream branch labelled **Frozen f_θ → linear classifier → cross-subject prediction**.

The figure is produced by `src/slices/george/figures.py` (project-tracked). The script writes `papers/kics-george/fig1-pipeline.pdf` (gitignored, local-only). Reference in LaTeX:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig1-pipeline.pdf}
\caption{Proposed pipeline. CSI samples are augmented twice with independently sampled Doppler warp factors; the two views are encoded and contrasted via NT-Xent. The pre-trained encoder is frozen for cross-subject linear-probe evaluation.}
\label{fig:pipeline}
\end{figure}
```

---

## 12. End-of-session protocol

When you reach a STOP point, or when you have completed all eight tracer bullets:

1. Make sure all branches are pushed.
2. Make sure all open PRs have a fresh comment summarising current state.
3. Comment on issue #34 (or the most recent open issue) with a summary of: which tracer bullets are done (with PR numbers), which is in progress (with branch name and current state), what the next AFK session — or George — needs to do to resume.
4. If T1.7 or T1.8 is in flight, also include the local path to the compiled paper PDF in the comment (so George can find it on the local machine).
5. Stop generating output. Wait for the human.

---

## 13. Quick reference: handy commands

```bash
# View an issue
gh issue view 34

# View all Slice 1 issues
gh issue list --label "slice:george-doppler"

# Open a PR with auto-fill from commit
gh pr create --fill --assignee Jupiter-Plantagenet

# Comment on an issue
gh issue comment 34 --body "Status update: ..."

# View open PRs
gh pr list --state open

# Diff between current branch and main
git diff main..HEAD --stat

# Run the slice's smoke test
pytest tests/slices/george/

# Compile the paper (T1.7+; runs in local-only papers/kics-george/)
cd papers/kics-george && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 14. Final note for the AFK Claude session

You have authority to make decisions inside any one tracer bullet — code style, file naming, helper functions, plot aesthetics. You do **not** have authority over: the experimental design (what augmentations, what dataset, what split), the paper's scientific claims, or the submission itself. Stay inside the lane this document defines. When in doubt, comment on the relevant issue and stop.

Good luck. The paper is winnable.
