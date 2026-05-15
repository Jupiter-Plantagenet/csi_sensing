# Slice 6 — Composability of Doppler + Coherence-Mask (Victor)

Per `docs/08-team-work-plan.md` §5 and `docs/09-execution-roadmap.md` §2.2.

**Theme.** Decompose a raw-CSI sample into a *static* component (the slowly-varying contribution from walls, furniture, ceiling — the room) and a *dynamic* component (the fast-varying contribution from the moving person). Replace the static component with one from a different sample to simulate "the same activity in a different room." Tested on cross-subject + interaction term, the most direct test of room-invariance.

Run the smoke test:

```bash
python -m src.slices.victor.run
pytest tests/slices/victor/
```

T6.1 (this PR) ships the scaffold on stub data. Subsequent tracer-bullets:

- **T6.2** real Widar3.0 raw-CSI cross-subject + interaction term loader (train/test from different recording dates → different rooms).
- **T6.3** reimplementation of Doppler + coherence-mask (default: temporal lowpass at 2 Hz).
- **T6.4** decomposition sanity test (synthetic slow + fast components).
- **T6.5** composability study augmentation (run each augmentation individually + combined).
- **T6.6** generic baseline + static-perturbation comparison, single seed.
- **T6.7** multi-seed comparison.
- **T6.8** 1-page writeup at `papers/team/composability.md`.

The deliverable is a figure showing cross-subject + interaction term accuracy for the static-perturbation row alongside the hand-crafted baseline row (T5.6) in `papers/team/comparison-figure.png`.
