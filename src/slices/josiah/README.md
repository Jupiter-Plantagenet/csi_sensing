# Slice 5 — Baseline Reproduction (Josiah)

Per `docs/08-team-work-plan.md` §8 and `docs/09-execution-roadmap.md` §1, this slice produces the team paper's "comparison to conventional solutions" — five anchored baselines on Widar3.0 cross-subject. T5.1 (this tracer-bullet) ships the supervised-no-SSL scaffold on stub data so the pipeline runs end-to-end before real-data work begins in T5.2.

Run the smoke test:

```bash
python -m src.slices.josiah.run
pytest tests/slices/josiah/
```

The printed accuracy after T5.1 is at chance level — stub data has random labels. T5.2 replaces the stub with a real Widar3.0 cross-subject loader and the same `run.py` produces a defensible accuracy.

## What each file does

- `data.py` — `StubCSI` for plumbing (T5.1). `Widar3CrossSubject` lands in T5.2.
- `encoder.py` — `TinyCNN` encoder (same architecture as Slice 1's encoder, reimplemented here per the slice-independence rule) and `SupervisedClassifier` (encoder + linear classifier head).
- `eval.py` — `train_supervised` (cross-entropy training) and `evaluate` (top-1 accuracy).
- `run.py` — entrypoint stringing it all together.

## Baselines this slice ships (over its 8 tracer-bullets)

1. **T5.2** — Supervised, no SSL. Real cross-subject data, 3 seeds.
2. **T5.3** — SimCLR with trivial augmentation (random crop only).
3. **T5.4** — Exact AutoFi reproduction (Yang et al. 2022). 6-layer Conv2d (Table I), twin GSS branches, L = L_p + λL_m + γL_g with λ=1, γ=1000 (eq. 9), SGD lr=0.01 momentum 0.9 for 300 epochs. Module: `autofi.py`. Run: `python -m src.slices.josiah.run --mode autofi`.
4. **T5.5** — Exact CAPC reproduction (Barahimi et al. 2024). Twin RSCNet-style encoders + GRU autoregressive heads + per-step bilinear predictors; hybrid loss L = L_BT + β(L_CPC^A + L_CPC^B) with β=50, λ_BT=0.002 (eq. 6); LARS optimiser, 300 epochs. Module: `capc.py`. Run: `python -m src.slices.josiah.run --mode capc`. Note: paper's `dual view` augmentation needs uplink+downlink CSI; Widar3.0 ships only one direction, so we use the documented CAPC* noise+subcarrier-mask fallback (Table I row).
5. **T5.6** — Hand-crafted-aug SimCLR (Gaussian noise + random subcarrier mask). **The comparison column for slices 1, 2, 4, 6.**

T5.7 builds `papers/team/baselines-figure.png` from the five baseline runs; T5.8 writes the gap-analysis paragraph.
