# Slice 4 — Coherence-Aware Subcarrier Masking (Ihunanya)

Per `docs/08-team-work-plan.md` §5 and `docs/09-execution-roadmap.md` §2.2.

**Theme.** Decompose a raw-CSI sample into a *static* component (the slowly-varying contribution from walls, furniture, ceiling — the room) and a *dynamic* component (the fast-varying contribution from the moving person). Replace the static component with one from a different sample to simulate "the same activity in a different room." Tested on cross-subject (and robustness sweep), the most direct test of room-invariance.

Run the smoke test:

```bash
python -m src.slices.ihunanya.run
pytest tests/slices/ihunanya/
```

T4.1 (this PR) ships the scaffold on stub data. Subsequent tracer-bullets:

- **T4.2** real Widar3.0 raw-CSI cross-subject (and robustness sweep) loader (train/test from different recording dates → different rooms).
- **T4.3** coherence-bandwidth estimation (default: temporal lowpass at 2 Hz).
- **T4.4** decomposition sanity test (synthetic slow + fast components).
- **T4.5** coherence-aware subcarrier masking augmentation (block-mask subcarriers matching coherence bandwidth).
- **T4.6** generic baseline + static-perturbation comparison, single seed.
- **T4.7** multi-seed comparison.
- **T4.8** 1-page writeup at `papers/team/coherence.md`.

The deliverable is a figure showing cross-subject (and robustness sweep) accuracy for the static-perturbation row alongside the hand-crafted baseline row (T5.6) in `papers/team/comparison-figure.png`.
