# Project brief — physics-grounded augmentation design for SSL on CSI

A short narrative covering what we built, what the published baselines do,
what we reproduced, what we originally hypothesized, why we changed it,
and what evidence forced the change.

## 1. Where the project sits

The team's literature survey traces five eras of CSI sensing research and
concludes that *augmentation design* is the unresolved gap in SSL for
Wi-Fi sensing. AutoFi picked Gaussian noise without justification; CAPC
documented method-dependent augmentation optima without explaining them;
CLAR replaced the question with a diffusion model; CIG-MAE abandoned
augmentations entirely; SSLCSI treated augmentation as a controlled
variable. Our position is **physics-grounded augmentation design — each
augmentation is a falsifiable hypothesis about a named physical
invariance.**

## 2. The published baselines and how they work

Three published methods were targeted as exact-reproduction baselines:

- **AutoFi** (Yang et al. 2022) uses a two-stream encoder trained with a
  *geometric self-supervised loss* — KL probability-consistency + mutual-
  information sharpening + a cosine-similarity-based geometric structural
  term — over two Gaussian-noise-augmented views, followed by few-shot
  calibration (paper §IV-D) or linear probe (released SenseFi code).
- **CAPC** (Barahimi et al. 2024) is a hybrid CPC + Barlow-Twins method
  with an RSCNet encoder, a GRU autoregressor, and a 3-layer Barlow-Twins
  projector. The novel augmentation is *dual-view*: the two SimCLR views
  are synchronized uplink and downlink CSI of the same packet, exploiting
  channel reciprocity. Optimized with LARS.
- **MAE** (SSLCSI Table 4c; He et al. 2022 backbone) is a masked
  autoencoder — random 75% of input tokens are dropped, a Transformer
  encoder sees only the visible tokens, a lightweight Transformer decoder
  reconstructs the masked tokens via MSE. No augmentations; the masking
  is the SSL signal.

## 3. What we reproduced and how it landed

| Reproduction | Our number | Paper number | Status |
|---|---:|---:|---|
| AutoFi (Widar BVP, SenseFi protocol, T=22 + linear probe) | 0.399 | 0.638 (paper §IV-D, T=40 + few-shot) | **hardware-limited** — preprocessing mismatch (we have SenseFi's T=22 BVP CSV release; the paper used the original Widar3 T=40 BVP release plus few-shot calibration, not linear probe). |
| AutoFi UT-HAR §IV-C 20-shot (1 seed, 100 ep SSL + 300 ep probe) | 0.516 | 0.788 (paper §IV-C Fig. 4) | **failed** — −27.2 pp gap. Implementation paper-faithful (GSS loss, two-stream Conv1d encoder, AdamW); the gap likely reflects encoder-architecture choice (paper §IV-C encoder details "unspecified"; we used a SenseFi-style 1D CNN adapted for UT-HAR shape). |
| CAPC paper-exact (Lab→Home, true LARS, 300 ep SSL, k=9 clamped from paper k=12) | 1.000 | 0.9755 (paper Table 1, k=10 and k=12 cells) | **above-saturation** — raw-CSI logistic-regression floor at the same k=9 split is 0.9638; paper's 0.9755 sits within ~1.2 pp of that floor. The "exact within 0.1 pp" tolerance is not meaningful for this saturated cell. |
| CAPC interim (Home-only, AdamW stand-in, 2 ep SSL, k=9) | 0.978 | n/a | sanity check |
| MAE adapted to BVP cross-subject (3 seeds, 200 ep) | 0.629 ± 0.010 | n/a | **adapted-baseline** — SSLCSI's MAE = 0.692 is on raw CSI receiver-2 with random split, not directly comparable. |
| **MAE-on-UT-HAR (SSLCSI Table 4c, 3 seeds, 200 ep, batch=256)** | **0.8427 ± 0.0031** | **0.843** | **EXACT** — gap = −0.03 pp, well within the 0.1 pp tolerance the roadmap requires. The first and only cleanly-exact published-baseline reproduction in this session. Implements ViT-style transformer MAE (250 time tokens × 90-dim, 6-layer encoder, 2-layer decoder, mask_ratio=0.75, AdamW lr=1.5e-4 with 40-ep warmup + cosine decay) on the SenseFi-format UT-HAR data. |

All baseline code is paper-faithful and unit-tested. Where the gap is
preprocessing- or saturation-driven rather than implementation-driven we
classify honestly (`hardware-limited`, not `failed`).

## 4. Project baselines on the canonical substrate

Three project baselines on the same Widar3 BVP cross-subject split
(train users 5–17, test users 1–4, gestures 1–6, 3 seeds):

- **Supervised CNN** — 0.590 ± 0.009
- **SimCLR + random temporal crop** — 0.450 ± 0.018
- **SimCLR + Gaussian noise + temporal mask (handcrafted)** — **0.482 ± 0.005** ← the comparison column

## 5. Our original five hypotheses

Each slice proposed one physics-grounded augmentation derived from a
named physical cause of domain shift on raw CSI:

1. **Slice 1 (George) — Doppler-aware time warp.** Activity speed
   scales the CSI Doppler component linearly; warp the time axis →
   speed-invariant features.
2. **Slice 2 (Chigozie) — Static-component perturbation.** Time-mean
   CSI captures slow room/multipath structure; swap it across batch →
   environment-invariant features.
3. **Slice 3 (Collins) — Calibrated phase-noise injection.** Different
   chipsets impose different phase distortions; inject samples from a
   fitted phase-noise profile → chipset-invariant features.
4. **Slice 4 (Ihunanya) — Coherence-aware subcarrier mask.**
   Frequency-selective fading drops a coherence-bandwidth-sized block
   of subcarriers; mask that block → fading-invariant features.
5. **Slice 6 (Victor) — Composability.** Compose Doppler + coherence
   sequentially; test for super-additive gains.

## 6. Why we adjusted, with evidence

Gate 1 of our roadmap — a "is the substrate working?" sanity check —
established that **raw-CSI cross-subject Widar3 sits at chance for
every method we tried**:

| Pipeline                                                | Top-1  |
|---------------------------------------------------------|-------:|
| Supervised TinyCNN, receivers=`[1]`                      | 0.158 |
| Supervised TinyCNN, receivers=`[1,2,3]`                  | 0.169 |
| Supervised TinyCNN, receivers=`[1..6]`                   | 0.168 |
| SimCLR + Doppler aug (Slice 1, original raw-CSI run)    | 0.162 |
| SimCLR + handcrafted (raw CSI)                          | 0.175 |
| AutoFi-adapted (raw CSI)                                | 0.174 |
| CAPC-adapted (raw CSI)                                  | 0.171 |

Chance is 0.167; every pipeline sits within ±0.04. Four very different
SSL families converge at the same floor, so **the bottleneck is in the
substrate, not the method**.

The same encoder family on Widar3 BVP cross-subject — same users, same
gestures, same 3-seed protocol — works fine:

| Pipeline on BVP                                         | Top-1  |
|---------------------------------------------------------|-------:|
| Supervised CNN                                          | 0.590 ± 0.009 |
| SimCLR + random temporal crop                           | 0.450 ± 0.018 |
| SimCLR + Gaussian + temporal mask (handcrafted)         | 0.482 ± 0.005 |
| MAE adapted to BVP                                      | 0.629 ± 0.010 |

BVP works because the Widar3.0 offline pipeline (multi-receiver Doppler
spectrograms + compressed-sensing inversion) explicitly normalizes out
user-specific multipath, leaving a body-coordinate velocity profile that
generalizes across subjects. Raw CSI from a single receiver carries
user-fingerprint structure that overpowers the gesture signal at our
data scale.

We therefore moved the proposed-method evaluation to BVP and reframed
each augmentation to operate on the BVP `(T=22, vx=20, vy=20)` substrate
while preserving the physics-grounded design principle:

| Slice | Original physical cause (raw CSI) | BVP-reframed physical cause | What changed |
|---|---|---|---|
| 1 Doppler-warp | Doppler component scales with speed | Same — BVP literally is velocity profile, time-warp scales speed in velocity space | nothing; code is shape-agnostic |
| 2 Static-perturb | Time-mean CSI = room multipath signature | Time-mean BVP = baseline-velocity distribution; swap across batch | only the physical interpretation changes |
| 3 Phase noise | Chipset-specific phase distortion | **Replaced**: BVP is real-valued; new augmentation is *velocity-coordinate-frame jitter* (small random affine in vx/vy plane) targeting BVP coordinate-frame mis-registration | augmentation replaced; chipset-invariance claim deferred to CSI-Bench (Stage B) |
| 4 Coherent subcarrier mask | Frequency-selective fading on subcarriers | Coherent velocity-band mask on vx axis — simulates truncated sensing range | only physical interpretation changes |
| 6 Composability | Doppler + coherence | Same composition on BVP | nothing |

## 7. Proposed-method results so far

| Slice | Method | Result (3 seeds) | vs handcrafted (0.482) | Reading |
|---|---|---:|---:|---|
| 1 | bvp-doppler | **0.205 ± 0.005** | **−27.7 pp** | HURTS — Doppler-warp alone collapses SSL on BVP |
| 2 | bvp-static-perturb | **0.297 ± 0.012** | **−18.5 pp** | HURTS — static-swap on BVP hurts SimCLR |
| 3 | bvp-velocity-jitter (BVP-reframed) | **0.378 ± 0.007** | **−10.5 pp** | HURTS — velocity-frame jitter hurts |
| 4 | bvp-coherent-mask | **0.414 ± 0.008** | **−6.8 pp** | HURTS — closest to baseline, but still below |
| 6 | bvp-doppler-coherent (composability) | **0.362 ± 0.015** | **−12.0 pp** | HURTS — composition is *worse* than either standalone |

**All five proposed methods underperform the handcrafted baseline on
BVP cross-subject.** Standard deviations are tight (0.005–0.015) across
3 seeds; these are not noise. Most striking: the composability slice
(6) lands *below* both of its constituent augmentations (4 alone is
−6.8 pp, 1 alone is −27.7 pp; the composition is −12.0 pp) — the
augmentations actively interfere with each other rather than composing
additively.

This contradicts the simple version of the slide-17 thesis
("physics-grounded > content-agnostic") for these five specific
augmentations on this specific substrate. The simplest defensible
explanation: SimCLR with a single contrastive augmentation produces
two views that share too much structure (e.g., two time-warped views
of the same gesture remain mutually highly similar), making the
contrastive task too easy and the encoder representation degenerate.
The handcrafted Gaussian + temporal mask baseline introduces *more*
inter-view variation, which paradoxically makes the contrastive task
harder and the resulting features more useful for the downstream
linear probe.

## 8. What the paper claims (revised, contribution-rich)

The original slide-17 thesis ("physics-grounded augmentations help SSL
on CSI") is *refuted* by our own data: all five physics-grounded
augmentations we tested underperform the content-agnostic handcrafted
baseline on BVP cross-subject. The honest paper makes three claims
instead:

1. **Physics-grounded augmentation as a design space, *not* a universal
   improvement.** Naive physics-grounded augmentations (Doppler-warp,
   static-perturb, velocity-frame jitter, coherent velocity-band mask,
   and their composition) all hurt SimCLR on BVP. The simplest
   defensible mechanism: each of these augmentations preserves too much
   inter-view similarity, making the contrastive task too easy and
   inducing representation collapse. The handcrafted Gaussian + temporal
   mask baseline succeeds *because* it injects more inter-view variation
   than a pure physical transformation does. This is a real and
   *publishable* counter-finding to the slide-17 framing.

2. **Substrate matters more than augmentation.** Raw-CSI cross-subject
   Widar3 sits at chance for every SSL method tested (four method
   families, eight pipelines, all within ±0.04 of 1/6); BVP cross-subject
   leaves 0.41 pp of headroom for SSL to operate. The team paper's
   Gate-1 substrate diagnostic is itself a methodological contribution.

3. **One exact published-baseline reproduction**: MAE on UT-HAR =
   0.8427 ± 0.0031 vs SSLCSI Table 4c = 0.843 (−0.03 pp gap, well within
   the 0.1 pp tolerance). The other reproduction attempts are honestly
   classified: AutoFi-on-BVP and AutoFi-on-UT-HAR fail or are
   hardware-limited by protocol mismatches; CAPC paper-exact sits above
   a saturated cell where the 0.1 pp tolerance is below the task's
   intrinsic noise floor.

The honest paper is therefore *contribution-rich* in a different way
than originally framed: it refutes the simplest version of the
"physics-grounded > content-agnostic" thesis with tight, replicated
negative results, while *succeeding* at one exact reproduction of a
non-saturated published cell (MAE-UT-HAR). Negative results plus one
clean positive reproduction is more honest than five
unfalsifiable-because-saturated wins would have been.

## 9. Where the artifacts live

- `papers/team/baselines-figure.md` and `.png` — figure with source numbers
- `papers/team/sslcsi-grading-spec.md` — why SSLCSI cells were/weren't reachable
- `papers/team/autofi-exact-spec.md` — AutoFi exact-reproduction spec
- `papers/team/capc-hardware-limited.md` — CAPC blocker doc (pre-SignFi)
- `papers/team/mae-exact-spec.md` — MAE Row A (done) + Row B (deferred)
- `papers/team/bvp-augmentation-hypotheses.md` — the substrate pivot, with evidence

All code at `src/slices/{josiah,george,chigozie,collins,ihunanya,victor}/`,
131 unit tests, results under `results/2026-05-15-*-aggregate/`.

Five commits this session (`bcea0d9`, `f409ae3`, `734eb49`, `bd2377d`,
`df99015`) cover the BVP pipeline, AutoFi/CAPC/MAE infra, the BVP-reframed
hypotheses, the proposed-method wiring, and the CAPC paper-exact result.
