# Slice 1 T1.5 — doppler real-data cross-subject (first real run)

**Result.** Linear-probe accuracy 0.162 vs chance 0.167. Doppler-aug SSL pre-training does not separate from chance on the cross-subject split, single seed.

**SSL convergence.** Loss 3.035 → 2.8967 across 300 epochs; most of the drop happens by epoch ~10 then plateaus. NT-Xent on 6 classes has a theoretical floor around log(6) ≈ 1.79, so the model is far from collapse — but also far from sharp class separation in the projection space.

**What this means for the paper.** This is one data point at one seed. Don't draw conclusions yet. Need:
- Hand-crafted SimCLR comparison at the same seed (T5.6 column) — next run.
- 3-seed variance on whichever rows look meaningful.
- Cross-environment split as a second test (Slice 2 already wires this up).

**Caveats from this run.**
- `receivers=[1]` only (one of six redundant receiver views per gesture). Bounds CPU memory at ~1GB peak. May lose representational diversity from the other angles — re-running with `receivers=[1,2,3]` once we have lazy loading would be worth comparing.
- `time_steps=100` is a fixed crop/pad; real Widar3.0 packets have variable length (typically 1200-1800). Aggressive crop, but consistent across all baselines.
- `linear_probe accuracy on stub data: 0.162` printout label is a Slice 1 T1.1 leftover string; the data is real.

**Provenance.** See `git_hash.txt` for code state; `config.yaml` for full hyperparameters; `stdout.log` for the full run log.
