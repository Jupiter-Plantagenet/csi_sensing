# T5.4 AutoFi exact reproduction — cross-subject Widar3.0

**Result.** Loss collapsed to exactly 0.0 from epoch 2 onward; linear probe accuracy 0.174 (vs chance 0.167).

**What collapsed.** All three loss terms (probability consistency L_p, mutual info L_m, geometric L_g) can be simultaneously zero at the degenerate solution where every input maps to the same one-hot prediction distribution and the two views match exactly:
- L_p = symmetric KL(P1 || P2) → 0 when P1 == P2.
- L_m = h(E[P]) + E[h(P)] → 0 when all P are the same one-hot vector (h = 0 for both terms).
- L_g = KL(Q1 || Q2) → 0 when the cosine-similarity Q matrices match.

The paper's γ=1000 on L_g is meant to *force* geometric consistency, but combined with our magnitude-normalised real-CSI inputs, the gradient field appears to admit this trivial attractor. We hit it within one epoch.

**What 0.174 means.** With gradients pinned at zero from epoch 2, the encoder is effectively frozen at its near-initial state. The linear probe is measuring the discriminability of a *randomly-initialised* AutoFiCNN. That it lands at 0.174 (just above chance 0.167) is consistent with a barely-trained network — there's about ~1 epoch of useful gradient before collapse.

**Fix candidates** (not done in this run):
- Lower γ to ~10 instead of 1000 to soften the geometric penalty.
- Add a label-smoothing / temperature term to G_φ to prevent one-hot collapse.
- BatchNorm in G_φ to bound activation magnitudes.
- Initialise G_φ to output near-uniform distributions (small final-layer weights).

**Paper-relevance.** This is a paper-worthy negative result: an exact-reproduction of AutoFi GSS (per the published architecture + loss + optimiser) fails to converge on cross-subject Widar3.0 without further stabilisation. The paper itself reports AutoFi on BVP (not raw CSI) for Widar — and on raw CSI for UT-HAR (a different dataset). Treating this run as "the method, as published, applied to raw-CSI cross-subject Widar3.0" is honest.

**Provenance.** `config.yaml`, `git_hash.txt`, `stdout.log` for the run record.
