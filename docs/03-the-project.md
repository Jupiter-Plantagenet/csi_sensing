# 03 — The Project

This document describes what we are actually building. It assumes you've read the previous two — you know what CSI is, what SSL is, what augmentation is, and roughly what's wrong with how augmentation is currently chosen. If any of those are still fuzzy, go back.

By the end you should be able to explain, to someone new, exactly what our project does, why each of our four augmentations is chosen, and how we plan to test them.

---

## 1. The research question, in one sentence

**Which augmentations help a self-supervised CSI encoder generalize across rooms, hardware, and people — and can we explain *why* by tying each augmentation to a specific physical phenomenon?**

Read that sentence twice. It contains everything.

- *Augmentations* — the load-bearing concept from the previous doc.
- *Self-supervised CSI encoder* — the object we are training.
- *Generalize across rooms, hardware, people* — the three kinds of variation that make CSI sensing fail in the real world.
- *Explain why* — we want understanding, not just numbers. The "why" is what separates this from a benchmark study.
- *Specific physical phenomenon* — our hypothesis is that real-world variation has identifiable physical causes, and we can design augmentations that simulate each cause.

---

## 2. Why generalization is the right target

A model that scores 99% on the dataset it was trained on, and 60% in a different room, is not useful. WiFi sensing will only become a deployed technology when models work in environments their developers have never seen. This is called **domain generalization** in the ML literature, and it is the most important practical limitation of CSI sensing today.

There are three main kinds of domain shift in CSI:

- **Cross-environment** — the model was trained in Room A but is now used in Room B. The walls, furniture, geometry are different, so the static multipath structure is different.
- **Cross-hardware** — the model was trained on one WiFi chipset (say, Intel 5300) but is now used on another (say, Broadcom or ESP32). Different chipsets introduce different phase calibration offsets and may have different numbers of subcarriers.
- **Cross-subject (or cross-speed)** — the model was trained with one set of people but is now used on different people. Different people have different gaits, body sizes, movement speeds.

Each of these is a real, measurable problem, and each has a clear physical cause. That is what makes them tractable.

---

## 3. The core idea — causal ablation

Most studies of augmentation just ablate accuracy: "we tried augmentation X, and accuracy went up by 2%." This is shallow. Concrete example: if augmentation X improves accuracy by 2% in Room A but you don't know *why*, you cannot predict whether X will help in Room B, on a different chipset, or with a different downstream task. Each new deployment becomes another guess. Without the *why*, the field accumulates results that don't compose into knowledge.

We're going to do something different. For each augmentation we test, we make a specific claim about *what kind of generalization it should improve* — and then we test it on the *specific kind of domain shift* it's supposed to help with.

The unit of work is one row of this table:

| Augmentation | The physical phenomenon it simulates | Hypothesis | Dataset / split it's tested on |
|---|---|---|---|
| Static-component perturbation | Environment change → static multipath profile changes | Pre-training with this aug → better cross-environment accuracy | Widar3.0 cross-environment split |
| Calibrated phase noise injection | Hardware swap → different phase calibration | Pre-training with this aug → better cross-chipset accuracy | CSI-Bench cross-chipset split |
| Coherence-aware subcarrier masking | Frequency-selective fading varies | Pre-training with this aug → better robustness to held-out subcarriers and to cross-environment evaluation | Widar3.0 + held-out subcarrier robustness |
| Doppler-aware time warping | Activity speed varies → Doppler scales | Pre-training with this aug → better cross-subject accuracy (subjects vary in speed) | Widar3.0 cross-subject split |

Each row is a *falsifiable hypothesis*. We can be wrong, and the experiment will tell us so. That's the point.

---

## 4. The four augmentations, explained

Here's what each augmentation actually does and why.

### 4.1 Static-component perturbation

CSI from a room at any moment can be approximately split into two parts:

- A **static component**, which is the contribution from things that aren't moving — the walls, the ceiling, the table, the unused chair.
- A **dynamic component**, which is the contribution from things that *are* moving — most importantly, the human whose activity we want to recognize.

If you move from Room A to Room B and a person performs the same activity, the static component changes (different walls, different furniture) but the dynamic component, in some sense, stays similar (the person did the same thing).

**The augmentation:** decompose a CSI sample into static and dynamic components, replace the static component with one from a different sample, and recombine. The resulting augmented sample is "the same activity in a different room." Pre-training with this teaches the encoder that the static component is irrelevant to identity.

**The risk:** static-vs-dynamic decomposition is hard to do cleanly. The simplest method (temporal lowpass filter) can leak slow body motions into the "static" estimate. We need a working decomposition before this augmentation can be tested. It is the highest-risk of the four to implement.

### 4.2 Calibrated phase noise injection

Different WiFi chipsets — the actual silicon inside a router or laptop — introduce different small distortions to the signals they receive. Each chipset has its own characteristic profile: tiny imperfections in its internal clock, small variations in when it samples the signal, slight unpredictability in when it decides a packet has begun. These show up as random-looking shifts in the *phase* of the CSI, even when the wireless channel itself hasn't changed. Common WiFi chipset families used in CSI research include the Intel 5300, the Broadcom BCM43xx series, and the ESP32; each leaves its own fingerprint, and a model trained on one usually performs worse on another.

**The augmentation:** measure each chipset's phase noise profile from data (CSI-Bench has data from five chipset families, which is exactly what we need). During pre-training, randomly sample a chipset profile and inject the corresponding phase noise into a CSI sample. The augmented sample looks like "this CSI but as it would have been measured on a different chipset."

Pre-training with this teaches the encoder that hardware-specific phase distortions are irrelevant.

**The risk:** "calibrated" requires actually measuring chipset-specific phase noise profiles, which is its own preprocessing pipeline before we can do augmentation. This is moderate effort but well-defined.

### 4.3 Coherence-aware subcarrier masking

Recall: WiFi splits its signal into 30–256 narrow subcarriers, and each subcarrier is affected by the channel slightly differently. **Frequency-selective fading** is what we call it when some subcarriers come through strong while others are weakened or cancelled, depending on how reflections happen to add up at each frequency. But adjacent subcarriers, ones that are close in frequency, are *correlated* — they fade together. The width of this correlation is called the **coherence bandwidth** and is determined by how spread out the multipath is in time.

When existing methods mask subcarriers, they pick subcarriers at random. This destroys the coherence structure — like erasing random pixels from an image instead of erasing a coherent patch.

**The augmentation:** instead of random subcarrier masking, mask a contiguous block of subcarriers whose width matches the coherence bandwidth. This simulates a realistic frequency-selective fading event — a portion of the spectrum dropping out together, the way it would in a real noisy channel.

Pre-training with this teaches the encoder that block-wise subcarrier dropouts are noise to be ignored, not signal.

**The risk:** the right block size depends on the channel. Too narrow and we're back to random masking; too wide and we destroy the activity signal. We'll need to estimate coherence bandwidth from data and pick a block size that matches.

### 4.4 Doppler-aware time warping

Walking faster causes a larger Doppler shift on the multipath components that bounce off the walker. The relationship is approximately linear: double the speed, double the Doppler.

**The augmentation:** stretch or compress the time axis of a CSI sample by a random factor — say, between 0.7× and 1.4×. This simulates the same activity performed at different speeds. The Doppler signatures shift accordingly.

Pre-training with this teaches the encoder that activity speed is a degree of freedom, not an identity-defining feature.

**The risk:** this is the simplest augmentation of the four to implement, but it's also the most approximate. A real person walking faster doesn't just have higher Doppler — their gait pattern actually changes. We're modeling speed as a linear time-axis stretch, which is a first-order approximation. Whether this approximation is good enough is itself an experimental question.

---

## 5. The experimental plan

### 5.1 Datasets

| Dataset | What it gives us | Cross-domain splits available |
|---|---|---|
| **Widar3.0** | 6 gestures, 17 users, 3 rooms. The most-used WiFi sensing benchmark. | Cross-environment, cross-subject, cross-orientation |
| **CSI-Bench** | Multi-task data from 5 different chipset families, in-the-wild conditions. Available on Kaggle. | Cross-chipset (the only public dataset that gives us this) |
| **UT-HAR** | Activity recognition data from a single environment. | Within-domain only — we use it for label-fraction studies, not cross-domain claims. |

### 5.2 SSL methods

We use **two** methods for the core ablation:

- **SimCLR** — the canonical contrastive method. Strongly augmentation-dependent.
- **Barlow Twins** — a non-contrastive method. Also augmentation-dependent, but the math works differently.

Why two methods? Because CAPC found that augmentation effects depend on the method. If a physics-informed augmentation helps SimCLR but hurts Barlow Twins, that's an important finding — it tells us the augmentation interacts with the method, not with CSI.

If compute budget allows, we may also include **MAE** (masked reconstruction) as a single comparison run. This addresses CIG-MAE's argument that augmentations are unhelpful — without doing a full ablation on it.

### 5.3 The protocol

For each (augmentation, method, dataset) combination:

1. **Pre-train** the encoder on unlabeled CSI from the source domain using that augmentation.
2. **Fine-tune** with a small amount of labeled data from the target domain (we sweep label fractions: 1%, 5%, 10%, 50%, 100%).
3. **Test** on a held-out target-domain test set.
4. **Repeat** with at least 3 random seeds, report mean and standard deviation.

We always include a **baseline run** with the same SSL method but generic augmentations (Gaussian noise + random subcarrier masking, the most common defaults). This is what each physics-informed augmentation has to beat.

### 5.4 What success looks like

A successful project produces a table like this:

| Augmentation | Cross-env Δ accuracy | Cross-chipset Δ accuracy | Cross-subject Δ accuracy |
|---|---|---|---|
| Static-component perturbation | **+X%** ✓ | small | small |
| Phase noise injection | small | **+Y%** ✓ | small |
| Coherence-aware masking | medium | medium | small |
| Doppler-aware warping | small | small | **+Z%** ✓ |

Where each augmentation specifically helps the kind of generalization it was *predicted* to help with — and doesn't help the others as much. That's the causal claim. If we get a result like that, we have evidence that the physics-to-augmentation mapping is real.

Even a *negative* result is useful. If physics-informed augmentations don't beat baselines, that's a finding the field needs.

---

## 6. Scope discipline

The most common way class projects fail is by trying to do too much. To prevent that, here is what we are explicitly **not** doing.

- We are not designing a new SSL method. We use SimCLR and Barlow Twins exactly as published.
- We are not designing a new neural architecture. We use the same encoder backbones as the AutoFi and CAPC papers.
- We are not collecting new CSI data. We use Widar3.0, CSI-Bench, and UT-HAR as published.
- We are not trying to set new accuracy records on the benchmarks. Our goal is *understanding which augmentations work and why*, not absolute accuracy.

Scope-creep is the enemy. If someone has an idea that doesn't fit the table in section 3, we write it down for "future work" and move on.

---

## 7. Open questions we still need to answer

These are real things we have not decided yet. Some will become specific GitHub issues.

1. **What's our static/dynamic decomposition method?** Probably a temporal lowpass filter at first, but the cutoff frequency needs to be chosen, and we need a sanity check that the dynamic component still classifies activities at near-baseline accuracy.

2. **How do we fit chipset-specific phase noise distributions from CSI-Bench?** This is its own preprocessing pipeline.

3. **How do we estimate coherence bandwidth from data?** Coherence bandwidth can be derived from the **channel impulse response** — a representation that tells you how the channel responds over time after a brief input — but we need a concrete recipe for computing it from CSI.

4. **What's the minimum viable experiment we run first?** Probably one augmentation × one method × one dataset, to make sure the whole pipeline works end-to-end before scaling.

5. **What backbone encoder do we use?** AutoFi and CAPC use slightly different backbones. We need to pick one and stick with it.

6. **What's the labeled-data fine-tuning protocol?** Linear probe on frozen features, or full fine-tuning? Different baseline papers report different things in different setups, and we need to pick a primary protocol and stick with it.

These open questions are not flaws in the plan. They are the things engineering work will resolve. The next document describes how we coordinate that engineering work.

Continue to [`04-github-workflow.md`](04-github-workflow.md).

---

## Glossary of terms introduced in this doc

| Term | Meaning |
|---|---|
| Domain generalization | The ML problem of building models that work on data from environments not seen during training. |
| Cross-environment / cross-hardware / cross-subject | The three kinds of domain shift relevant to CSI sensing. |
| Causal ablation | An experimental design where each variable is mapped to a hypothesis about cause, and tested against a matching outcome. |
| Static component | The part of CSI from things that aren't moving in the environment. |
| Dynamic component | The part of CSI from things that are moving (typically humans). |
| Coherence bandwidth | The frequency width over which the channel's effect on different subcarriers is correlated. |
| Linear probe | Fine-tuning protocol where the encoder is frozen and only a single linear classifier on top is trained. |
| Full fine-tuning | Fine-tuning protocol where the encoder is also updated alongside the head. |
| Widar3.0 / CSI-Bench / UT-HAR | The three public CSI datasets we will use. |
