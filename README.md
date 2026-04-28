# Physics-Informed Augmentations for Self-Supervised CSI Sensing

A class research project for *Special Topics in Advanced Signal Processing* (Smart Factories track).
We are studying which data augmentations help neural networks learn good representations of WiFi Channel State Information (CSI), and why.

---

## What this project is, in one paragraph

Modern WiFi can do more than carry data — the same signals can be used to sense the environment, detecting things like human activity, gestures, or falls without cameras. This works because as a WiFi signal bounces around a room, it picks up a fingerprint of everything it touches. That fingerprint, called Channel State Information (CSI), can be fed to a neural network that learns to recognize what's happening in the room. The catch: training such networks normally requires huge amounts of carefully labeled CSI, which is expensive to collect. *Self-supervised learning* (SSL) lets the network learn from unlabeled CSI by training on artificially modified — *augmented* — versions of each sample. **Our project is a careful study of what augmentations work for CSI, why, and which physical phenomena each one is really teaching the model about.**

We test four physics-informed augmentations specifically:

- **Static-component perturbation** — simulates a room change.
- **Calibrated phase noise injection** — simulates a hardware change.
- **Coherence-aware subcarrier masking** — simulates realistic frequency-selective fading.
- **Doppler-aware time warping** — simulates a person performing the same activity at a different speed.

Each is described in detail in [`docs/03-the-project.md`](docs/03-the-project.md).

---

## Who is on this team

| Name | Student ID |
|---|---|
| George Chidera Akor | 20246163 |
| Victor Ikenna Kanu | 20246166 |
| Chigozie Athanasius Nnadiekwe | 20246125 |
| Ihunanya Udodiri Ajakwe | 20245055 |
| Josiah Ayoola Isong | 2025210186 |
| Collins Izuchukwu Okafor | 20246171 |

---

## How to read these docs

If you are new to this whole area, read the docs in order. Each one assumes you've read the previous one.

| Order | Doc | What it covers | Time |
|---|---|---|---|
| 1 | [`docs/01-foundations.md`](docs/01-foundations.md) | What wireless signals, machine learning, and deep learning are. Written for someone with no background in either. | ~45 min |
| 2 | [`docs/02-csi-and-ssl.md`](docs/02-csi-and-ssl.md) | What CSI is, what self-supervised learning is, what augmentation is, and how the field got to where it is today. | ~45 min |
| 3 | [`docs/03-the-project.md`](docs/03-the-project.md) | What our specific project is, the four augmentations we're studying, and how we'll test them. | ~30 min |
| 4 | [`docs/04-github-workflow.md`](docs/04-github-workflow.md) | How we coordinate work on this repo: issues, branches, pull requests, reviews. | ~20 min |
| 5 | [`docs/05-first-week-checklist.md`](docs/05-first-week-checklist.md) | Step-by-step walkthrough of your first hour and your first pull request. Read this once you've been added to the repo. | ~20 min |
| 6 | [`docs/06-using-ai-well.md`](docs/06-using-ai-well.md) | How to use AI assistants well on this project: as a research assistant, not an oracle. Read before posting on any design issue. | ~15 min |

If you only have **30 minutes**, read this README and `docs/03-the-project.md`. You'll be able to follow conversations and contribute to discussions, even if some of the technical context is fuzzy.

If you only have **5 minutes**, read just this README. You'll know what we're doing and where to look for more.

---

## What we are *not* doing

This is just as important as what we are doing. Scope discipline keeps a class project finishable.

- We are **not** building a new neural-network architecture.
- We are **not** building a new self-supervised learning method.
- We are **not** collecting new CSI data.
- We are **not** trying to beat published accuracy records.

We are doing **one specific thing**: running a careful, controlled study of how a small set of augmentations — designed to mirror real physical phenomena that affect wireless signals — change what an SSL model learns about CSI.

---

## Status

This document was last updated on **April 28, 2026**. The project has produced three class presentations to date (in `*.pptx`) and is now moving into the implementation phase. No code has been written yet. The first concrete engineering milestone is reproducing one baseline (likely AutoFi or CAPC) on one dataset (likely Widar3.0 or CSI-Bench).

---

## Where to ask questions

If something in any of these docs is unclear, that is a bug in the docs. Open an issue using the `area:docs` label (see [`docs/04-github-workflow.md`](docs/04-github-workflow.md)) and we'll fix it. The goal is for someone joining the team in week 6 to understand the project as well as someone who has been here from week 1.
