# Slice 1 T1.5 — handcrafted real-data cross-subject (comparison row)

**Result.** Linear-probe accuracy 0.175 vs chance 0.167. Hand-crafted SimCLR (Gaussian noise + random subcarrier mask) marginally above chance on the cross-subject split, single seed.

**Side-by-side with doppler (same seed/epochs/data):**

| Augmentation | SSL final loss | Linear-probe acc |
|---|---|---|
| `doppler_warp` | 2.897 | 0.162 |
| `gaussian_then_mask` | 2.923 | 0.175 |

Hand-crafted edges Doppler by 1.3 percentage points; neither is meaningfully separating from chance (0.167 on 6 classes).

**Honest read.** TinyCNN at 40k params + SimCLR at temperature 0.5 + linear probe is collectively *weak* on the cross-subject 17-user split. The encoder is undersized relative to dataset diversity, and the probe is too restrictive given the high-variance feature space SimCLR produces here. Don't conclude "doppler doesn't help" from this run alone — the architecture is likely the bottleneck. AutoFi's 2.4M-param Conv2d encoder (T5.4) should be the next comparison; it has the capacity to potentially exceed chance noticeably.

**What this is worth.** First real cross-subject data point with the team paper's intended pipeline. Establishes a working baseline that other runs can be compared to. Don't include in the team paper without (a) handcrafted multi-seed variance, (b) larger encoder (AutoFi/CAPC), (c) cross-environment as a second figure axis.
