# Slice 2 — Static-Component Perturbation (Chigozie)

Per `docs/08-team-work-plan.md` §5 and `docs/09-execution-roadmap.md` §2.2.

**Theme.** Decompose a raw-CSI sample into a *static* component (the slowly-varying contribution from walls, furniture, ceiling — the room) and a *dynamic* component (the fast-varying contribution from the moving person). Replace the static component with one from a different sample to simulate "the same activity in a different room." Tested on cross-environment, the most direct test of room-invariance.

Run the smoke test:

```bash
python -m src.slices.chigozie.run
pytest tests/slices/chigozie/
```

T2.1 (this PR) ships the scaffold on stub data. Subsequent tracer-bullets:

- **T2.2** real Widar3.0 raw-CSI cross-environment loader (train/test from different recording dates → different rooms).
- **T2.3** static/dynamic decomposition (default: temporal lowpass at 2 Hz).
- **T2.4** decomposition sanity test (synthetic slow + fast components).
- **T2.5** static-component perturbation augmentation (swap statics across batch).
- **T2.6** generic baseline + static-perturbation comparison, single seed.
- **T2.7** multi-seed comparison.
- **T2.8** 1-page writeup at `papers/team/static.md`.

The deliverable is a figure showing cross-environment accuracy for the static-perturbation row alongside the hand-crafted baseline row (T5.6) in `papers/team/comparison-figure.png`.
