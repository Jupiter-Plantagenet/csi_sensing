# Slice 1 — George — Doppler-aware time warping

This directory holds the end-to-end SSL pipeline for the Doppler-aware
time-warping slice. See `docs/08-team-work-plan.md` section 4 for the slice
roadmap and `docs/07-experiment-scaffold.md` for project-wide defaults.

## Layout

| File | What it holds |
|---|---|
| `encoder.py` | `TinyCNN` (~40K params for Widar3.0 input shape) and the (T,S,A)→(C,T) reshape helper. |
| `ssl.py` | SimCLR wrapper, projection head, NT-Xent loss, pre-training loop. |
| `eval.py` | Linear-probe evaluation on a frozen encoder (sklearn LogisticRegression). |
| `data.py` | Dataset wrappers. T1.1 ships `StubCSI`; T1.2 adds the real Widar3.0 loader. |
| `run.py` | End-to-end entrypoint that wires the pieces together. |

## Run the smoke test

```bash
python -m src.slices.george.run
pytest tests/slices/george/
```

Both should complete without error. The accuracy printed by `run.py` on stub
data is not meaningful — see issue #34.

## Status

- T1.1 — scaffold + SimCLR end-to-end with stub data — implemented in this PR.
- T1.2 onward — see issues #35–#41.
