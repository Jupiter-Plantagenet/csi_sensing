# 2026-05-15: cross-subject raw-CSI on Widar3.0 sits at chance

**Finding.** Six runs on the same cross-subject split (train users 5-17, test users 1-4, receivers=[1], magnitude-only, time_steps=100 unless noted) all land within ±2 points of chance (0.167 on 6 classes):

| Method | Encoder | Epochs | T | Linear-probe / top-1 acc |
|---|---|---|---|---|
| Supervised | TinyCNN 40k | 50 | 100 | 0.155 |
| Supervised | TinyCNN 40k | 50 | 200 | 0.183 |
| SimCLR + doppler aug | TinyCNN 40k | 300 | 100 | 0.162 |
| SimCLR + gaussian+submask aug | TinyCNN 40k | 300 | 100 | 0.175 |
| AutoFi (collapsed at ep 2) | AutoFiCNN 2.4M | 300 | 100 | 0.174 |
| CAPC | RSCNet-approx + GRU 549k | 300 | 100 | 0.171 |

**What this says.** This isn't a single-method failure. Methods of four very different shapes (supervised, SimCLR, AutoFi GSS, CAPC hybrid) all converge to ~chance on the same data. The bottleneck is in the preprocessing → split pipeline, not the SSL methodology.

**Likely causes (ordered).**
1. **Magnitude-only projection** in `csi_complex_to_real` drops the phase signal. Widar3.0 gestures' Doppler information is split across real/imag; magnitude alone may not be enough.
2. **`receivers=[1]`** single viewing angle, while the published Widar3 / AutoFi pipelines either fuse all 6 receivers (BVP) or use them as independent samples.
3. **Cross-subject domain gap is intrinsically hard** on raw CSI — AutoFi itself reports its Widar3 numbers on BVP (body-coordinate velocity profile), not raw CSI.

**Status.** Halted by user. Next-session decisions on file: pivot to BVP loader (most likely matches published baselines), or extend the raw-CSI pipeline (phase + all receivers).

**Provenance.** Individual run metrics + configs under `results/2026-05-15-*-real-*` and `results/2026-05-15-josiah-T5.2-supervised-T200/`.
