# Slice 1 — KICS paper status

This is a project-tracked status pointer for George's KICS paper. The paper
artefacts themselves (`main.tex`, `refs.bib`, `main.pdf`,
`fig1-pipeline.pdf`, `SUBMISSION.md`) live under the gitignored
`papers/kics-george/` directory on the AFK execution machine; they never
appear in any PR diff. See the AFK plan's commit-boundary rationale at
`docs/slice-1-afk-plan.md` section 1.

## Draft state (2026-05-07)

- **Local PDF**: `papers/kics-george/main.pdf` (2 pages, IEEEtran two-column,
  198 KB).
- **LaTeX source**: `papers/kics-george/main.tex` — full structure (abstract,
  IEEE keywords, intro, methodology with two subsections, results table,
  conclusion, acknowledgment, references). Uses the project-wide
  acknowledgment block verbatim from the prior paper.
- **References**: `papers/kics-george/refs.bib` — 7 entries (AutoFi, CAPC,
  SimCLR, Widar3.0, Xu et al. SSL benchmark, two of George's prior works).
  Each entry compiles cleanly via `bibtex`; substantive verification (open
  the actual paper, confirm it says what the citation claims) is a T1.8
  step per AFK plan section 7.8 / `docs/06-using-ai-well.md`.
- **Figure 1**: rendered by `src/slices/george/figures.py` (project-tracked).
  Output `papers/kics-george/fig1-pipeline.pdf` (gitignored). Style mirrors
  the prior paper's Fig. 1 — left-to-right pipeline with Widar3.0 input,
  two Doppler-warped views, encoder, NT-Xent loss, and a frozen-encoder
  linear-probe branch. Regenerate with `python -m src.slices.george.figures`.
- **Result numbers**: placeholders (`TBD ± TBD`) in the abstract and Table I.
  Substituted in T1.8 once T1.6's multi-seed comparison produces real
  numbers. The data access blocker on issue #35 is the upstream gate.

## Compile from scratch

```bash
python -m src.slices.george.figures        # writes papers/kics-george/fig1-pipeline.pdf
cd papers/kics-george
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`main.pdf` should land at exactly 2 pages.

## Notes for the reviewer

- Title carries "Towards" framing; drop it in T1.8 if the result clears the
  convention rule (`mean(ours) > mean(baseline) + std(baseline)`).
- The discussion paragraph and the conclusion's closing sentence are
  marked with bracketed `[insert ...]` notes for T1.8 to flesh out once
  the result sign is known. The negative-result protocol (AFK plan §11)
  is already accommodated: only the title, abstract last sentence, and
  discussion paragraph need to change.
- The acknowledgment block matches the prior paper's wording verbatim
  (see AFK plan section 11.3); please don't re-edit unless the funding
  lines have changed.
- No content from `papers/kics-george/` appears in this PR's diff. The
  reviewer should pull the branch and run the compile-from-scratch
  recipe above to inspect the actual PDF.

## Status

- [x] T1.7 — paper draft compiles cleanly, 2 pages, all citations resolved.
- [ ] T1.8 — references verified one-by-one, result numbers substituted,
      submission portal upload checklist (`papers/kics-george/SUBMISSION.md`,
      gitignored) drafted, submission staged for George's manual portal
      upload.
