# Slice 3 — Collins — Calibrated phase-noise injection

This directory holds the end-to-end SSL pipeline for the calibrated
phase-noise-injection slice. See `docs/08-team-work-plan.md` section 6 for the
slice roadmap and `docs/07-experiment-scaffold.md` for project-wide defaults.

## Slice plan, in one paragraph

Different WiFi chipsets introduce different phase distortions. The slice fits a
phase-noise profile from real CSI, then synthesises "what would this CSI look
like under different phase noise" by injecting samples from the profile during
SimCLR pre-training. **Stage A** (this branch and the immediately following
ones) prototypes the augmentation on Widar3.0 raw CSI (Intel 5300, single
chipset) and reports cross-subject robustness. **Stage B** (follow-up issues
once CSI-Bench is downloaded) fits per-chipset profiles and reports cross-
chipset transfer — the slice's headline causal claim.

## Layout

| File | What it holds |
|---|---|
| `encoder.py` | `TinyCNN` (~40K params for Widar3.0 input shape) and the (T,S,A)→(C,T) reshape helper. |
| `ssl.py` | SimCLR wrapper, projection head, NT-Xent loss, pre-training loop. |
| `eval.py` | Linear-probe evaluation on a frozen encoder (sklearn LogisticRegression). |
| `data.py` | Dataset wrappers. T3.1 ships `StubCSI`; T3.2 adds the real Widar3.0 loader. |
| `run.py` | End-to-end entrypoint that wires the pieces together. |

Future siblings (one per upcoming tracer bullet):

| File | Lands in |
|---|---|
| `phase_profile.py` | T3.3 (#52) — per-subcarrier Gaussian fit on phase residuals. |
| `augmentations.py` | T3.5 (#54) — generic baseline + phase-noise injection. |

## Run the smoke test

```bash
python -m src.slices.collins.run
pytest tests/slices/collins/
```

Both should complete without error. The accuracy printed by `run.py` on stub
data is not meaningful — see issue #50.

## Status

- T3.1 — scaffold + SimCLR end-to-end with stub data — implemented in this PR.
- T3.2 onward — see issues #51–#57.
