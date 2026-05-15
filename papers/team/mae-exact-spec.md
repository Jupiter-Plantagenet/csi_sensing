# MAE-for-CSI exact spec

Spec for the two MAE rows we plan to ship.

## Row A: `mae` on Widar BVP cross-subject (adapted, already implemented)

This is the row that goes in the team's baselines figure alongside
`bvp-supervised`, `bvp-simclr-trivial`, `bvp-simclr-handcrafted`, `autofi`.

* **Input:** `[T=22, vx=20, vy=20]` BVP from SenseFi `Widardata/`
  (normalization `(x - 0.0025) / 0.0119`).
* **Tokenization:** each time step = one token. 22 tokens per sample,
  token-dim 400 (the flattened 20x20 spatial slice).
* **Masking:** random 75% per sample, paper default (He et al. 2022).
* **Encoder:** 4-layer pre-norm Transformer, emb_dim 128, 4 heads, MLP
  ratio 4. Sinusoidal positional embedding.
* **Decoder:** 2-layer pre-norm Transformer at decoder_dim 64, learned mask
  token, sinusoidal positional embedding, linear head to token-dim 400.
* **Loss:** MSE on masked positions only (paper Eq. 1).
* **Optimizer:** AdamW lr=1.5e-4, wd=0.05, 40-epoch linear warmup + cosine
  decay over 200 epochs total.
* **Linear probe:** scikit-learn `LogisticRegression` on mean-pooled
  encoder features (matches `linear_probe_bvp` in `bvp_methods.py`).
* **Code:** `src/slices/josiah/mae.py`.
* **Tests:** `tests/slices/josiah/test_mae.py` (9 tests).
* **Smoke:** 2-epoch run on the cross-subject split returned
  linear-probe accuracy 0.378 — 2.3× the 6-class chance level.
* **Classification:** `adapted-baseline`. Not graded against a published
  cell (SSLCSI's 69.24% is on raw CSI receiver-2 with random split — a
  fundamentally different protocol; see
  `papers/team/sslcsi-grading-spec.md`).

## Row B: `mae-raw-r2` on Widar raw CSI (SSLCSI protocol, deferred)

This is the row that *would* be graded against SSLCSI's MAE/ViT Widar_R2
= 69.24% cell to ±0.1 pp.

**Not implemented as of 2026-05-15.** Requires:

1. A new raw-CSI loader for `[T, 30, Na=3]` amplitude-only, receiver=2,
   all 17 users, 22 activities, mean-pooled along time to `T=500`, random
   0.6/0.2/0.2 train/val/test split.
2. A ViT-Base encoder (12 layers, 768 hidden, 12 heads, MLP ratio 4).
3. MMSelfSup MAE defaults: mask_ratio 0.75, decoder 8x512x16, AdamW.
4. LR / warmup / batch from the SSLCSI grid; exact winning combo is in the
   GitHub repo (`JJJinx/SSLCSI`):
   * `SSLCSI/configs/selfsup/_base_/models/mae_vit-base.py` — model.
   * `SSLCSI/configs/selfsup/mae/mae_vit-base_*_widar*.py` — schedule.
   * `SSLCSI/configs/_base_/datasets/widar*.py` — data pipeline, patch size.
   * `SSLCSI/models/algorithms/mae.py` — masking impl.

Estimated effort:

* Loader + tests: ~1.5 hours.
* ViT-Base encoder + MMSelfSup-compatible config: ~1.5 hours.
* Wiring `mae-raw-r2` method into `production_runner`: ~30 min.
* Training one seed to first-pass convergence on CPU: ~2 hours.

If Phase 1 (MMSelfSup defaults, no grid search) misses the 0.1 pp tolerance,
Phase 2 is grid-searching the LR / warmup / batch combinations the SSLCSI
authors searched — a separate ~144-config sweep at ~30 min each is
prohibitive on CPU. Phase 2 should run on GPU only.

## Extra rows reachable with the same ViT encoder

Once `mae-raw-r2` exists, three additional SSLCSI rows are one-head-swap away
(same encoder, different SSL head):

| New method            | Target cell           |
|-----------------------|----------------------:|
| `moco-vit-r2`         | 62.81% (Table 4c)     |
| `simclr-vit-r2`       | not reported, project |
| `supervised-vit-r2`   | 86.33% (Table 4c)     |

All on the same Widar_R2 raw-CSI random-split protocol.
