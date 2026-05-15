# T5.5 CAPC exact reproduction — cross-subject Widar3.0

**Result.** Loss decreased monotonically 314 → 13.6 over 300 epochs (no collapse, no NaN). Linear-probe accuracy 0.171 vs chance 0.167.

**Behaviour was healthy.** CAPC's hybrid loss (L_BT + 50·(L_CPC^A + L_CPC^B)) trains stably — the CPC term provides hard negatives that prevent the AutoFi-style trivial-fixed-point collapse. So this is what a *correctly-trained* SSL method produces on this task.

**Why it still barely beats chance.** Concatenated per-window embeddings → logistic regression doesn't transfer across subjects. The learned representations capture *something* (loss decreased 23x), but it's not gesture-class discriminative information that survives the subject shift. This is consistent with the AutoFi paper's own choice to evaluate Widar3 on **BVP** (body-coordinate velocity profile) rather than raw CSI — BVP discards subject-dependent components by design.

**Pattern across four runs (same split, same seed).** Slice 1 doppler 0.162, Slice 1 handcrafted 0.175, T5.4 AutoFi 0.174, T5.5 CAPC 0.171. Spread 1.3 percentage points, all clustering at chance. Strongly suggests the bottleneck is *the split*, not the methods.

**Next sanity check (queued).** Run the supervised baseline on the same split. If supervised also lands near chance, the issue is the preprocessing (time_steps=100 crop, receivers=[1], magnitude-only) or the cross-subject split definition itself, not the SSL methods.

**Provenance.** `config.yaml`, `git_hash.txt`, `stdout.log`.
