# SSLCSI grading spec

**Paper:** Xu, Wang, Zhu, Zheng. *Evaluating SSL for WiFi CSI-Based HAR: A
Systematic Study.* ACM TOSN 21(2), 2025. arXiv:2308.02412.
**Code:** https://github.com/JJJinx/SSLCSI

This is the spec we tried to grade our AutoFi-on-BVP and MAE-on-BVP rows
against. **The protocol mismatch is fundamental** — SSLCSI runs on raw CSI
amplitude with a random 60/20/20 split; our BVP baselines run on the
SenseFi-format BVP CSVs with a cross-subject split. The two cannot be
graded against each other to ±0.1 pp.

## What SSLCSI actually evaluates

| Encoder | UT-HAR | SignFi | Widar_R2 |
|---|---|---|---|
| Supervised CausalNet | 98.03 | 83.33 | 85.48 |
| SimCLR ResNet        | 83.93 | 47.10 | 40.43 |
| MoCo ResNet          | 82.14 | 46.74 | 41.13 |
| SwAV ResNet          | 87.50 | 57.43 | 36.39 |
| Rel-Pos ResNet       | 82.14 | 81.34 | 32.18 |
| MoCo ViT             | 73.21 | 87.14 | 62.81 |
| **MAE ViT**          | 84.29 | 88.77 | **69.24** |

Table 4 / Section 4.2 of the arXiv version. All-percent.

**Critical:** AutoFi is **not** in the SSLCSI evaluation. Table 1 of SSLCSI
categorizes AutoFi as "Instance Discrimination" but the actual benchmark
runs only SwAV, SimCLR, MoCo, Rel-Pos, and MAE (Table 3, p.8).

## Widar_R2 ≠ our cross-subject BVP split

**Widar_R2** (Table 2 of SSLCSI, p.7):

- All rooms × **receiver-position 2** × all 17 users × **22 activities** × 45,170 records.
- Single-receiver subset, not a room split. "R2" = receiver index 2.
- Random **0.6 / 0.2 / 0.2 train / val / test split**.
- Inputs: `[T=500, 30, 3]` **amplitude only** (or `[500, 30, 6]` amp+phase),
  CSI mean-pooled along time to `T=500`.
- 22-class.

**Our cross-subject BVP split** (used by `bvp-supervised`,
`bvp-simclr-trivial`, `bvp-simclr-handcrafted`, `mae`):

- BVP from SenseFi release (`Widardata/`) × **gestures 1-6** × users 5–17
  train / 1–4 test × 24,388 records.
- Cross-subject (no user appears in both splits).
- Inputs: `[T=22, vx=20, vy=20]` BVP velocity profile.
- 6-class.

Different input modality, different number of classes, different
generalization regime. Reproducing 69.24% to ±0.1 pp on our split is
impossible by definition.

## Implication for grading our rows

| Our row | Grade against | Status |
|---|---|---|
| `bvp-supervised` (0.590 ± 0.009) | Widar3.0 paper TPAMI 2022 cross-domain supervised CNN-on-BVP (Zhang et al. 2022), if a cross-user cell exists. Else this is a self-published project baseline. | grading source TBD |
| `bvp-simclr-trivial` (0.450 ± 0.018) | SSLCSI's SimCLR-ResNet on Widar_R2 = 40.43% is **not** comparable (raw CSI, R2 single receiver, random split vs our BVP cross-subject). Closest published number is SenseFi benchmark SimCLR-on-Widar — also a different protocol. | **project-baseline** |
| `bvp-simclr-handcrafted` (0.482 ± 0.005) | Same as above. | **project-baseline** |
| `autofi` (running) | Paper §IV-D 0.638 (T=40 BVP + FSC) → hardware-limited by T=22 gap. SSLCSI has no AutoFi cell. | **hardware-limited** |
| `mae` (queued) | SSLCSI Widar_R2 = 0.692 (raw CSI, R2 receiver, random split) → **not** comparable; our MAE is adapted to BVP cross-subject. | **adapted-baseline**, not exact |
| CAPC | Paper Table 1 SignFi 0.9755 → hardware-limited (no SignFi UL/DL CSI). | **hardware-limited** |

## Two paths forward

### Path A (recommended given AFK constraints)

Treat MAE-on-BVP as an **adapted baseline** (not an exact reproduction).
Frame the figure as:

* 3 project baselines on BVP cross-subject (supervised + 2 SimCLR variants),
* 1 published reproduction attempt on BVP SenseFi-protocol (AutoFi,
  hardware-limited),
* 1 adapted MAE-on-BVP row (we adapted MAE to BVP, compares apples-to-apples
  with the SimCLR rows but is not a published-cell reproduction),
* 1 hardware-limited blocker doc (CAPC).

No exact published baselines, but the figure is internally consistent and
the published-baseline-attempt rows are honestly classified.

### Path B (Phase 2, if compute time permits)

Add a *second* MAE reproduction targeting SSLCSI's exact protocol:

* New loader for raw CSI amplitude-only, receiver-2, 22-class, mean-pooled
  to `T=500`, random 0.6/0.2/0.2 split (call it `widar-r2-random`).
* ViT-Base encoder (12 layers, 768 hidden, 12 heads) with patches
  scraped from `JJJinx/SSLCSI/configs/selfsup/_base_/models/mae_vit-base.py`.
* MMSelfSup defaults: mask_ratio 0.75, decoder 8×512, AdamW lr from the
  144-combo grid.
* Target: 69.24 ± 0.1 pp under the SSLCSI random split.

This is the only way to deliver an *exact* MAE reproduction. Adds a new
method `mae-raw-r2` to `production_runner` and a `WidarRawR2` loader.
Estimated implementation: ~4 hours. Estimated training: 1 seed × ~2 hours
on CPU.
