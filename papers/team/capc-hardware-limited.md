# CAPC reproduction: hardware-limited

**Method:** CAPC — Context-Aware Predictive Coding for Wi-Fi sensing.
**Reference:** Barahimi, Tabassum, Omer, Waqar. *Context-Aware Predictive
Coding: A Representation Learning Framework for WiFi Sensing.* IEEE OJ-COMS
2024. arXiv:2410.01825. Official code: `bornabr/CAPC`.

**Classification:** `hardware-limited` per
`docs/09-execution-roadmap.md` §1.1. CAPC's exact reproduction is **not
implementable on the data we have**.

## What the paper claims

CAPC reports headline numbers on **SignFi**:

| Cell                                                      | Number |
|-----------------------------------------------------------|-------:|
| SignFi-Home, linear eval, 12 shots/class (CAPC dual-view) | 97.55% |
| SignFi-Home, linear eval, 12 shots/class (CAPC* variant)  | 97.83% |
| SignFi-Home, linear eval, mean across 2–12 shots          | 89.82% |
| UT-HAR transfer, average                                  | 56.70% |

Source: Barahimi et al. 2024 §4.1.1 and Table 1; Table 2 for the UT-HAR
transfer cell.

## Why we cannot reproduce on this hardware

The paper's headline CAPC formulation is built around a **dual-view augmentation
that uses synchronized uplink and downlink CSI from the same packet** (§3.2,
§4.4). The view A / view B pair in the CPC + Barlow-Twins composite loss is
literally `UL` and `DL` CSI, not a pair of perturbed copies of one trace.

| Constraint                              | Status                          |
|-----------------------------------------|---------------------------------|
| Synchronized UL/DL CSI                  | SignFi only — **not** in Widar3.0 / UT-HAR. |
| SignFi dataset                          | Not present in `data/`.         |
| LARS optimizer                          | Not stock in PyTorch (workable but non-trivial). |
| RSCNet encoder channel widths           | Partially `[unspecified in paper]`; repo defaults required. |

Even on SignFi alone, our `data/` tree does not include it. We did not attempt
to download SignFi as part of the 2026-05-15 sprint: it is a multi-GB dataset,
the AFK constraint did not permit interactive remote-data setup, and the
project's primary baseline figure is on Widar3.0 BVP (see
`papers/team/baselines-figure.md`).

## What we shipped

`src/slices/josiah/capc.py` implements the method **end-to-end and tested**:

* `RSCNetEncoder` (per-window CNN, 3 residual `EncoderBlock`s).
* `CAPCAutoregressor` (GRU 128→128).
* `BarlowTwinsProjector` (3-layer MLP 128→256→256→256, bias-free output).
* `CAPCCPCLoss` (log-bilinear scorer with per-step `W_k`, in-batch negatives).
* `BarlowTwinsLoss` (cross-correlation with redundancy penalty `λ_bt = 0.002`).
* `CAPCLoss` composite (`L_BT + β·(L_CPC^A + L_CPC^B)` with `β = 50`).
* `build_capc_optimizer` (SGD with paper-faithful biases/BN LR split standing
  in for true LARS).
* `warmup_cosine_lr` (10-epoch linear warmup + cosine decay).
* Augmentations for the `CAPC*` single-view fallback: `capc_gaussian_noise`,
  `capc_time_flip`, `capc_subcarrier_mask`.

Tests at `tests/slices/josiah/test_capc.py` cover encoder/loss/projector
shapes, CPC loss finiteness, Barlow-Twins loss, the composite forward pass,
the optimizer split, and the warmup-cosine schedule.

## What is required to lift the hardware-limited flag

1. Obtain SignFi-Lab (5,520 instances, 276 classes; UL+DL CSI) and
   SignFi-Home (2,760 instances, 276 classes) from the SignFi authors.
2. Add a SignFi loader that returns `(view_a=UL, view_b=DL)` per sample.
3. Replace `build_capc_optimizer` with a true LARS implementation
   (`flax`-style or `lightning-bolts.optimizers.lars.LARS`).
4. Run with batch size 128 (paper) for 300 epochs SSL, then linear probe for
   100 epochs at LR 1e-2, k-shot eval on SignFi-Home.

Until then, any number we publish on CAPC must be labelled
`hardware-limited`, and the CAPC code path must not be exposed in the
`autofi`-style published-baseline rows of the figure.
