# AutoFi exact reproduction — spec

**Paper:** Yang, Chen, Zou, Wang, Xie. *AutoFi: Toward Automatic Wi-Fi Human
Sensing via Geometric Self-Supervised Learning.* IEEE IoT-J 2022.
arXiv:2205.01629. DOI 10.1109/JIOT.2022.3228820.

**Reference code:** `xyanchen/WiFi-CSI-Sensing-Benchmark` (SenseFi).
Relevant files: `self_supervised.py`, `self_supervised_model.py`,
`dataset.py`.

This is the spec our `src/slices/josiah/autofi.py` implements.

## Two distinct reproduction targets

| Source                | Task                                          | Headline number |
|-----------------------|-----------------------------------------------|----------------:|
| Paper §IV-D, Fig. 5   | Widar BVP, 16-cls SSL → 6-cls FSC, 20-shot   | **0.638** |
| Paper §IV-D, Fig. 5   | Widar BVP, 16-cls SSL → 6-cls FSC, 10-shot   | 0.556 |
| SenseFi released code | NTU-Fi HAR → HumanID transfer (14-cls)        | (paper §IV-B / Table II) |

We target the **paper §IV-D cell (0.638)** on the data we have: the SenseFi
processed Widar BVP CSVs (`Widardata/train` + `Widardata/test`).

## Inputs we use

* **Dataset:** Widar3.0 BVP (SenseFi release, `Widardata/`).
* **Per-sample tensor:** `(T=22, vx=20, vy=20)`. CSV rows = time, CSV cols =
  flattened 20×20 velocity grid. Normalization
  `x = (x − 0.0025) / 0.0119`.
* **Split (SenseFi protocol):** `Widardata/train` for SSL pre-training and
  linear-probe training; `Widardata/test` for evaluation. All 22 classes.
* **Known T-axis gap:** the paper §IV-D uses the original Widar3.0 BVP
  release with `T=40`. The SenseFi CSV release we have is downsampled to
  `T=22`. This is the dominant preprocessing-side reason an exact match to
  the paper's 0.638 is improbable.

## Encoder (per branch)

We adapt the SenseFi `CNN_encoder` (`self_supervised_model.py`) — designed
for NTU-Fi `(3, 114, 500)` input — to BVP `(22, 20, 20)`. The first kernel
sizes are changed to match the spatial input; the three-conv depth and the
projection→BN topology are preserved:

```
Input: (B, 22, 20, 20)
Conv2d(22 -> 32, k=3, padding=1)              ReLU
Conv2d(32 -> 64, k=3, stride=2, padding=1)    ReLU
Conv2d(64 -> 96, k=3, stride=2, padding=1)    ReLU
Reshape -> (B, 96*5*5 = 2400)
[unsupervised flag] Linear(2400 -> 256), BatchNorm1d(256)
[supervised   flag] return raw (B, 2400) features
```

Paper §IV-D explicitly says *"The first layer of the GSS module is slightly
modified to match the input size."* — the exact modified kernels are
`[unspecified in paper]`; our choice is the smallest faithful resize.

## Two-stream wrapper

Mirrors `CNN_Parrallel`:

```
encoder_1, encoder_2  — independent BVP encoders (no shared weights)
classifier            — Linear(2400 -> 128) ReLU Linear(128 -> num_classes)
forward(x1, x2, flag) — returns (encoder_1(x1, flag), encoder_2(x2, flag))
                        and classifier(...) when flag='supervised'
```

## SSL loss (GSS)

`AutoFiGSSLoss` returns `loss['final-kde']` exactly matching
`EntLoss::forward` in `self_supervised.py`:

```
probs_i      = softmax(feat_i)
sharpened_i  = softmax(feat_i / tau)

L_kl   = 0.5 * (KL(probs1 ‖ probs2) + KL(probs2 ‖ probs1))
EH     = 0.5 * (E[H(sharpened1)] + E[H(sharpened2)])
HE     = 0.5 * (H(E[sharpened1]) + H(E[sharpened2]))
L_kde  = cosine_similarity_loss(feat1, feat2)
final  = L_kl + (1+lam1)*EH − lam2*HE
total  = 100 * L_kde + final
```

Defaults (matching the released code argparse): `tau=1.0, eps=1e-5,
lam1=0.0, lam2=0.5, kde_weight=100`.

## Augmentation

`gaussian_noise_bvp(x, epsilon)` = `x + epsilon * N(mean=1, std=2)` —
per-call independent. View 1 uses `epsilon ~ U(0, 2.0)`; view 2 uses
`epsilon ~ U(0.1, 2.0)`.

## Optimizer and schedule (SenseFi protocol)

| Stage          | Optimizer                                 | LR    | Weight decay | Epochs |
|----------------|-------------------------------------------|------:|-------------:|-------:|
| SSL pre-train  | AdamW (all model params)                  | 1e-3 | 1.5e-6       | 100    |
| Linear probe   | Adam (`model.classifier.parameters()`)    | 1e-3 | 1e-5         | 300    |

No LR schedule. No warmup. Batch size 64 (our choice; SenseFi argparse
default is the same order of magnitude). `[unspecified in paper]`.

For the paper-§IV-D protocol (not what we run): SGD lr=0.01 momentum=0.9,
300 SSL epochs, batch 128, then 100 FSC fine-tune epochs of
`L_c + L_f` (cross-entropy + prototypical-net log-prob) on K-shot data
from 6 FSC classes.

## Reported number

`run_autofi(..., protocol="sensefi")` returns the *max-of-two-branches*
top-1 accuracy on the SenseFi test split, matching
`self_supervised.py::test`'s "Test accuracy: A, B" print.

## Classification rule

`src.production.classify_reproduction(reproduced, published=0.638)`:

* gap ≤ 0.001 → **exact**
* gap > 0.001 and any of {`T=22 ≠ T=40`, `linear probe ≠ FSC`,
  `class split unspecified`} hold → **hardware-limited**
* otherwise → **failed**

Even an in-tolerance match would honestly be reported with the protocol
caveats, because the released SenseFi code is the *only* faithful
reproduction surface for the gap-and-classification analysis.
