# T5.6 smoke training — within-subject plumbing validation

**Date:** 2026-05-15
**Configuration:** SimCLR pre-training with `gaussian_then_mask` augmentation (σ=0.05, p=0.15), `TinyCNN` encoder (~40K params), single-seed (42), 10 epochs, batch 32.
**Data:** Only `data/widar3/raw/20181128/user6/` extracted. Cross-subject split impossible with one user; using an 80/20 random split within user6 as a plumbing smoke.

## What I expected

- Pipeline runs to completion on real CSI without errors (loader → SimCLR → linear probe).
- SimCLR loss decreases monotonically over epochs.
- Linear-probe accuracy slightly above chance (chance = 1/6 ≈ 0.167) but not impressive, since we're training and testing on the same subject.

## What I saw

- 4500 samples loaded from user6.
- Pre-train losses (10 epochs): `[2.88, 2.58, 2.49, 2.45, 2.42, 2.41, 2.40, 2.38, 2.37, 2.37]` — monotonically decreasing.
- Linear-probe within-subject accuracy: **0.200** (vs chance 0.167).
- Total wall-clock: 54.3 s on CPU.

## Interpretation

- **Pipeline is sound.** Loader, csiread parsing, SimCLR training, linear probe — all run end-to-end on real Widar3.0 CSI without errors.
- **Accuracy is unimpressive but consistent with the setup.** Within-subject train/test on a single user is artificial; the encoder mainly has to learn user6's gesture variations from 4500 samples. A 3-pp improvement over chance after 10 epochs of SimCLR pre-training is a sanity check, not a result.
- **Cross-subject is not yet evaluable.** To produce the comparison-column number for the team paper, we need at least 4 users in the test set and the remaining 13 in train. That requires extracting more CSI archives.

## Next steps

1. Extract additional CSI archives (e.g. `CSI_20181117.zip`, `CSI_20181128.zip`) to get multi-user data.
2. Run the same configuration with the canonical cross-subject split (test on users 1–4).
3. Bump epochs to 50 for the production run; report mean ± std over 3 seeds.

The 3-seed cross-subject production run is what fills the `papers/team/comparison-figure.png` baseline row.
