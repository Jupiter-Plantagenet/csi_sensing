# 02 — CSI Sensing, Self-Supervised Learning, and the Augmentation Question

This document picks up where `01-foundations.md` left off. It assumes you already understand what a signal is, what multipath means, what a neural network does, and what self-supervised learning is at a high level. If those words don't ring a bell, go back to the previous doc.

By the end you should understand: what ISAC is, what CSI looks like as a piece of data, why machine learning is needed for CSI sensing, what self-supervised learning does for CSI specifically, what *augmentation* is and why it matters, and where the field currently is — including the four key papers that motivate our project.

---

## 1. ISAC — using one signal to do two jobs

Today, WiFi and radar are completely separate technologies, even though they both rely on radio waves. Your phone's WiFi connects you to the internet. Your car's radar warns you about pedestrians. Different hardware, different parts of the radio spectrum, no overlap.

This separation is wasteful. Radio spectrum is a finite, shared, increasingly crowded resource — and the same radio waves that carry data could in principle also be used to detect what's around. A WiFi signal already bounces off everything in the room before reaching your laptop. The bouncing pattern itself contains information about the room. Why throw it away?

**Integrated Sensing and Communication (ISAC)** is the idea of using one signal, on one piece of hardware, to do both jobs simultaneously. The signal carries data and its reflections are analyzed to sense the environment. ISAC is being officially built into 6G — the next generation of cellular networks. As of 2025, it is no longer a research curiosity. Major device manufacturers and standards bodies are building it.

For our purposes, you don't need to know much about ISAC. You only need to know that **CSI is the data ISAC produces**, and that anything you learn about doing useful things with CSI directly contributes to making ISAC work. Concretely: if ISAC sensing is going to ship in commercial 6G hardware, the models need to work across rooms, hardware, and people they have never seen before — and that is exactly the generalization problem augmentation is supposed to help with.

---

## 2. What CSI actually looks like

When a WiFi signal is sent, the receiver has to figure out how the channel distorted it before it can recover the original data. To do this, the transmitter sends a known reference pattern, and the receiver measures: "given that I know what was sent, and I see what I received, what must the channel have done in between?"

The answer to that question — for each subcarrier, for each pair of transmit and receive antennas, at each moment in time — is **CSI: Channel State Information**. It's a measurement of how the channel transformed the signal at each frequency.

### The shape of a CSI sample

A typical CSI sample, after being collected over a few seconds of WiFi traffic, looks like this:

```
shape: (time, subcarrier, antenna pair, complex value)
```

Concretely, you might have:
- **time:** 500 packets, captured at 100 packets per second over 5 seconds.
- **subcarrier:** 30 to 256 frequency sub-channels, depending on hardware.
- **antenna pair:** 1 to 9, if there are 3 transmit and 3 receive antennas.
- **complex value:** 2 numbers — the real part and the imaginary part — encoding amplitude and phase together. (See the "complex number" aside in `01-foundations.md` if this term feels new.)

So one CSI sample is a tensor of shape something like `(500, 56, 1, 2)` — a few hundred thousand numbers describing how the room affected a WiFi signal over five seconds.

### Why amplitude and phase together?

A wave has both a *strength* and a *position in its cycle*. Amplitude tells you how strong the arriving signal is on each subcarrier. Phase tells you the time-shift — equivalent, in physics terms, to the distance the signal traveled. When researchers talk about "complex CSI," they mean keeping both. Some papers throw phase away and use only amplitude. This is sometimes okay, but often loses information — phase carries geometric information about path lengths, which is exactly what you'd want for sensing.

### Why this is fundamentally different from images

A photo and a CSI sample look superficially similar — both are big tensors of numbers. But there are deep differences.

- An image's pixels are *spatial*: pixels next to each other correspond to physical points next to each other. CSI's "axes" mix time, frequency, and antenna geometry — three completely different physical things.
- An image's pixel values are real numbers between 0 and 1. CSI's values are complex numbers that can swing wildly across many orders of magnitude (the strength of an arriving WiFi signal can vary by a factor of a thousand or more).
- Two pixels in an image are independent in a simple way. Two CSI subcarriers can be coupled in subtle, structured ways — when reflections add up at a particular subcarrier, they may amplify it, while at a nearby subcarrier they cancel and weaken it. This effect is called **fading**, and we'll meet it again later.

This is going to matter when we talk about augmentation.

---

## 3. CSI sensing — how WiFi sees activity

A person standing still doesn't change CSI much. A person walking through the room changes CSI dramatically. As their arms and legs move, the multipath structure of the room shifts moment by moment — some reflection paths appear, others disappear, distances change. The CSI of "walking" looks different from the CSI of "sitting" looks different from the CSI of "falling."

You can train a neural network to recognize these patterns. Give it a chunk of CSI labeled "walking" and a chunk labeled "sitting," and it will learn to tell them apart. With enough labeled data and a well-designed network, accuracy on standard activity-recognition benchmarks reaches 95–99%.

This enables real applications. Fall detection for elderly people without cameras. Indoor localization where GPS doesn't work. Gesture recognition. Even breathing-rate monitoring, picking up the tiny CSI changes from a chest rising and falling. All of it works through walls, in the dark, without any new hardware beyond ordinary WiFi.

### The labeled data problem

The 95–99% accuracy numbers above all come from *supervised learning*. Someone painstakingly labeled the CSI: "this 5-second window is walking, this one is sitting, this one is gesturing." Doing this is expensive. Unlike image labeling (where a human can look at a photo and instantly say "cat"), nobody can look at a CSI tensor and say "walking." It's a matrix of complex numbers — incomprehensible to humans. To label CSI, you have to set up a controlled experiment: someone performs a specific activity, with a stopwatch, while CSI is recorded, and the labels are aligned afterward. This is slow, costly, and not easily scalable to dozens of environments and hundreds of subjects.

Meanwhile, every WiFi access point in the world is producing CSI 24 hours a day, completely unlabeled, and is throwing it away.

This is the gap self-supervised learning fills.

---

## 4. Self-supervised learning, in detail

Recall from the last document: SSL uses a pretext task to train a model on unlabeled data. After pretext-task training, the model has learned a useful encoder, and you only need a small labeled dataset to fine-tune a head for the actual task you care about.

There are several flavors of SSL. The two that matter most for our project are *contrastive learning* and *masked reconstruction*.

### Contrastive learning

The pretext task: given a CSI sample, produce two slightly different versions of it (by adding noise, masking parts of it, etc.). These are called the **two views**. Train the model so that the encoder produces *similar* representations for the two views of the same sample, and *different* representations for views of different samples. (Views from *other* samples that the model is told to push apart are called **negatives**.)

That's the whole idea. The model learns to ignore the kinds of changes that augmentation introduces — these become "things that don't matter for identity" — while keeping the parts that distinguish one sample from another. Why does this produce a useful representation? Because if two augmented views of the same input compress to the same point in the encoder's output, then the encoder has captured *what makes that input what it is*, separately from the kinds of variation the augmentation throws at it. Choose the right augmentations, and the encoder's notion of "same input" lines up with our notion of "same activity."

The most famous contrastive method is **SimCLR**. A close cousin, **Barlow Twins**, uses a slightly different mathematical trick to encourage similar things to compress together without needing explicit negatives — this makes Barlow Twins technically a *non-contrastive* method, even though it shares the same "two views" skeleton. **AutoFi** is a WiFi-specific contrastive method that we'll discuss below. The exact algorithmic differences between these methods are not important right now. What matters is that they all share the same skeleton: *augment, encode, compare*.

### Masked reconstruction

A different pretext task. Take a CSI sample, hide (mask) a chunk of it, and train the model to predict what was hidden using what's still visible. The most famous version is **MAE** (Masked Autoencoder), originally for images. An *autoencoder* is just an encoder with a decoder bolted on, trained to reproduce its own input — masking turns this into a "fill in the blanks" exercise the model can only solve if it has learned the structure of the data. Masked reconstruction does not need augmentation in the same way contrastive learning does — there are no "two views." But it has its own design question: *what to mask, and how much?*

We'll come back to this distinction. For now: contrastive learning depends on augmentation; masked reconstruction depends on masking. Both depend on a design choice that the field has not fully understood.

---

## 5. Augmentation — the load-bearing concept

In supervised learning, augmentation is a side concern. You augment to artificially expand the training set: rotate the cat photo, flip it horizontally, and you have a few more "cat" examples for free. It helps a little but isn't critical.

In self-supervised contrastive learning, augmentation *is* the training signal. The model learns specifically by seeing *what changes when you augment* and being told to ignore it. If you choose the wrong augmentations, you are teaching the model to ignore the wrong things. If you choose nearly identical augmentations, you give the model nothing to ignore, and it learns nothing.

### A worked example from images

In computer vision, researchers spent years figuring out which augmentations work for SSL on images. The answers turned out to be:

- **Random cropping:** showing the model two different cropped pieces of the same photo teaches it that "the identity of the object doesn't change when you look at a different part of it." This is a kind of *spatial invariance*.
- **Color jitter:** randomly perturbing the colors teaches the model that "the identity doesn't change under different lighting." This is *illumination invariance*.

Each augmentation encodes a specific invariance the researcher believed was useful. Crop teaches "ignore position." Color jitter teaches "ignore lighting." The augmentations are not arbitrary — they correspond to real-world variations that should not change what an image shows.

### Why augmentation for CSI is harder

Here is the problem. Most CSI researchers do not have an equivalent understanding for their data. They borrow augmentations from images (Gaussian noise, masking) without asking what real-world variation each one corresponds to. Sometimes the borrowed augmentations work; sometimes they don't; and nobody can fully explain the difference.

CSI is a physical measurement of electromagnetic propagation. There is, in principle, a deep theory connecting how a CSI sample changes when you change the room, the hardware, or the activity. That theory has not been used to design augmentations. It is the gap our project tries to fill.

---

## 6. The four-paper landscape

The field has produced a small number of very influential papers on SSL for WiFi sensing. We need to know four of them in particular, because our project's argument is built directly on what each one found.

### AutoFi (2023)

The first WiFi-specific SSL system. AutoFi takes a stream of unlabeled CSI from a new environment, runs a self-supervised pretext task to learn an encoder, and then fine-tunes the encoder on a small amount of labeled data for whatever task the user wants — gesture recognition, gait recognition, activity recognition. It works.

What's relevant for us: AutoFi uses two augmentations — adding random noise, and flipping the time axis. The paper does not justify *why* these two augmentations and not others. They tried some options and these worked best in their experiments. That's the whole reasoning.

This is normal practice in the field, and it's the practice we want to challenge.

### CAPC (2024)

The most important paper for our project. CAPC tested *six different augmentations* across *four different SSL methods*. The result is striking: **every method has a different optimal augmentation**. SimCLR works best with one combination. Barlow Twins prefers another. AutoFi prefers a third. The paper documents this finding clearly but doesn't explain why it happens.

CAPC also proposes one genuinely physics-grounded augmentation — using **uplink** (your phone sending to the router) and **downlink** (the router sending to your phone) CSI as two views of the same underlying channel, exploiting a property called **reciprocity** from electromagnetic theory which says the channel is the same in both directions. This is, as far as we know, the first physics-grounded augmentation in the literature.

The takeaway: augmentations matter enormously, and the relationship between augmentation choice and method choice is unexplained.

### CLAR (2024)

CLAR makes a sharp observation: when you apply standard image augmentations to CSI — Gaussian blur, for example — the augmented CSI looks almost identical to the original. The augmentation is too weak to give the model anything to learn from.

CLAR's solution is to give up on hand-crafted augmentations and learn them instead, using a generative model called a *diffusion model* trained specifically to produce diverse CSI variants. It works, but it's expensive, and you lose all interpretability — you can no longer say "this augmentation teaches that invariance," because the augmentation is now a black box.

The takeaway: generic augmentations are too weak for CSI, and learned augmentations work but at the cost of understanding.

### CIG-MAE (2025)

The most radical paper of the four. CIG-MAE argues that augmentations don't just fail for CSI — they actively destroy the physical information the model needs. Adding noise corrupts the subtle patterns that encode micro-movements. Masking subcarriers breaks the fading structure. Time-flipping reverses the Doppler signature, changing the physical meaning of the sample.

CIG-MAE's solution is to abandon augmentations entirely and switch to masked reconstruction. They mask large regions of the CSI tensor and train the model to fill them in. The system outperforms all the contrastive baselines.

The takeaway: there's a credible position in the field that says augmentations fundamentally don't fit CSI.

### What this means for our project

Four independent papers, each looking at a different angle, all point at augmentation as the unexamined lever. AutoFi shows it's not principled. CAPC shows the choice is method-dependent and unexplained. CLAR shows generic augmentations are too weak. CIG-MAE shows they may even be harmful.

Notice the tension this creates: AutoFi's two augmentations include time-flipping the CSI sequence, but CIG-MAE specifically points out that time-flipping reverses the Doppler signature, which changes the physical meaning of the sample. So one of the field's most-used augmentations is, by CIG-MAE's argument, unphysical. This kind of unresolved contradiction is exactly the symptom that motivates our project.

Our project takes a position: physics can guide augmentation design, and it's worth running a careful study to find out. We propose four physics-informed augmentations, each tied to one specific cause of domain shift in CSI sensing — *static-component perturbation* for room changes, *calibrated phase noise injection* for hardware changes, *coherence-aware subcarrier masking* for realistic frequency-selective fading, and *Doppler-aware time warping* for activity-speed variation. The next document explains each one in detail and lays out exactly how we'll test them.

Continue to [`03-the-project.md`](03-the-project.md).

---

## Glossary of terms introduced in this doc

| Term | Meaning |
|---|---|
| ISAC | Integrated Sensing and Communication. One radio signal doing both data and sensing. |
| CSI (Channel State Information) | Measurement of how the channel transformed a known reference signal. The data we work with. |
| Activity recognition | Classifying short windows of CSI into activity labels (walking, sitting, etc). |
| Localization | Estimating a person's position in space from CSI. |
| Pretext task | A task a model can solve without human labels, used to train an encoder in SSL. |
| Two views | The pair of augmented samples that contrastive SSL trains on. |
| Augmentation | A transformation applied to a sample to produce a variant for SSL training. |
| Invariance | The property of an encoder ignoring some kind of change in its input. |
| Contrastive learning | SSL via "pull together views of the same sample, push apart views of different samples." |
| Masked reconstruction | SSL via "hide part of the input and predict what was hidden." |
| SimCLR / Barlow Twins / MoCo / BYOL | Specific contrastive SSL methods. |
| MAE (Masked Autoencoder) | The standard masked-reconstruction method. |
| AutoFi | The first WiFi-specific SSL system, 2023. |
| CAPC | A 2024 SSL system whose augmentation analysis motivates much of our project. |
| CLAR | A 2024 system that uses learned augmentations from a diffusion model. |
| CIG-MAE | A 2025 system that abandons augmentations in favor of masked reconstruction. |
