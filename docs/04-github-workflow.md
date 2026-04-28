# 04 — How We Use GitHub for This Project

This document is for people who have heard of GitHub but have not used it for serious team coordination. By the end you should be able to find work, do the work without breaking what others are doing, and get your work reviewed and merged.

---

## 1. What GitHub is, in one paragraph

GitHub is a website that hosts code repositories ("repos") and adds collaboration tooling around them. The code itself is managed by a tool called **git**, which keeps a history of every change and lets multiple people work on the same code without overwriting each other. GitHub adds the social layer: issues to track work, pull requests to review changes before they get merged, and a project board to see what everyone is doing. Our project lives in one repo. Everything we build — code, configs, results, docs — goes in that repo.

If you have never installed git, do it now: [https://git-scm.com/downloads](https://git-scm.com/downloads). Make a GitHub account if you don't have one. Then ask George to add you to the team.

---

## 2. Getting set up

You'll need three things on your machine before any of these commands work: **git** (the version-control tool), **Python 3.10 or newer**, and **a way to run shell commands** — Terminal on macOS, Command Prompt or PowerShell on Windows, your normal shell on Linux. Install Python from [https://www.python.org/downloads/](https://www.python.org/downloads/) if you don't already have it.

You'll also need access to the repo on GitHub. The repo URL is something the team will give you (it'll look like `https://github.com/<our-org>/<our-repo>.git`); ask George if you don't have it. If you run `git clone` and get an authentication error, that means GitHub doesn't recognize you yet — ask George and we'll either add you to the organization or set up an SSH key with you.

```bash
# Clone the repo to your laptop (replace the URL with the real one)
git clone https://github.com/<our-org>/<our-repo>.git
cd <our-repo>

# Set up your name and email so commits attribute to you
git config user.name  "Your Name"
git config user.email "your-email@example.com"

# Install Python dependencies (when we have them)
pip install -r requirements.txt
```

That's it. You now have a local copy of everything in the repo. Anything you change here is local until you push it back.

---

## 3. The mental model

Imagine the project's code as a tree.

- The **trunk** is called `main`. It always works. It always passes tests. Nothing broken, ever.
- A **branch** is a copy of `main` you make to work on something specific. You can change anything you want on a branch without affecting `main`.
- A **commit** is a snapshot of the changes you've made. Commits stack up on a branch as you work.
- A **pull request** ("PR") is a proposal to merge your branch back into `main`. Other people review it, request changes, and approve it. Once approved, it's merged, and your work becomes part of `main`.

The whole point of this discipline is that `main` is never broken. If you discover the latest code on `main` doesn't work, that's a bug, and we fix it immediately. Everything else flows from this rule.

---

## 4. Finding what to work on

Look at the **Issues** tab on GitHub. Every piece of work is an issue.

Issues are labeled. The labels tell you what an issue is about and what kind of skill it needs.

| Label | What it means |
|---|---|
| `area:data` | Dataset loading, preprocessing, splits |
| `area:augmentation` | One of the four augmentations |
| `area:training` | Pre-training pipelines, fine-tuning, optimization |
| `area:evaluation` | Cross-domain protocols, metrics, statistical reporting |
| `area:baseline` | Reproducing AutoFi, CAPC, and other comparison systems |
| `area:docs` | Documentation, this README, the explainer |
| `area:infra` | Repo plumbing, CI, environment setup |
| `good-first-issue` | A small, well-scoped task suitable for starting out |
| `blocked` | Cannot start until another issue completes |
| `help-wanted` | Someone is stuck, please review |

To pick up an issue:

1. Find one with `good-first-issue` if you're new, or anything in your area otherwise. Make sure it isn't already assigned to someone else.
2. Comment on the issue saying "I'll take this." Assign yourself.
3. If you have questions, ask in the issue thread *before* starting. Better to clarify than to do the wrong thing.

---

## 5. Doing the work

### Starting a branch

Always make a new branch off the latest `main`. Never commit directly to `main`. **Always run `git pull` on `main` first** — otherwise your branch starts from a stale copy and you'll have to reconcile other people's work later.

```bash
# Make sure your local main is up to date — do this every time before starting a branch
git checkout main
git pull

# Create a new branch with a descriptive name
# (`git checkout -b NAME` creates a new branch and switches to it in one step)
git checkout -b <your-name>/<short-description>
# Examples:
#   git checkout -b george/static-dynamic-decomposition
#   git checkout -b victor/csi-bench-loader
```

The naming convention is `your-name/short-description`. This makes it obvious whose branch is whose.

If your branch lives for more than a few days, pull from `main` periodically while you're on the branch (`git pull origin main`) to keep it from drifting too far from where the rest of the team is.

### Committing

A **commit** is a snapshot of your changes plus a message explaining what they do. Make commits *small* and *focused*. A commit that does one thing is easier to understand, easier to review, and easier to revert if it breaks something.

```bash
# See what you've changed
git status

# Stage specific files for commit
git add path/to/file.py

# Commit with a message
git commit -m "Add temporal lowpass for static component estimation"
```

A good commit message:
- Starts with a verb in the imperative ("Add", "Fix", "Refactor", "Remove")
- Is a single sentence in the subject line, under 72 characters
- Explains *what* and *why*, not *how* (the diff shows how)

A bad commit message is "stuff" or "wip" or "asdf". Don't.

### Pushing your branch

```bash
git push -u origin <branch-name>
```

`origin` is git's default name for the remote copy of the repo on GitHub — when you cloned it, git automatically named that remote `origin`. The `-u` flag tells git to remember the connection between your local branch and the remote one, so future pushes can be just `git push`.

---

## 6. Pull requests

When your branch is ready for review — it doesn't have to be perfect, just ready for someone else to look at — open a pull request.

### Opening one

Go to the repo on GitHub, click "Compare & pull request" on your branch (or use the PR tab and create one manually). Fill in the description. Use this template:

```
## What this does
A 1–3 sentence explanation of what this PR changes.

## Why
Link to the issue: "Closes #42"

## How I tested it
What you ran, what output you saw, what you confirmed works.

## Anything reviewers should pay attention to
Open questions, places you're unsure, design decisions worth discussing.
```

### Getting review

PRs need at least one approval before they can merge. For anything touching shared infrastructure (the data loader, the training loop, evaluation code), get two approvals.

When someone reviews your PR, they'll either:
- **Approve** — looks good, merge it.
- **Request changes** — make these changes before merging.
- **Comment** — non-blocking thoughts.

Don't take review feedback personally. It is *always* about the code, not about you. Address each comment with either a code change or a reply explaining why you're not going to change it. Once all blocking feedback is resolved, the reviewer re-approves and you can merge.

### Reviewing someone else's PR

Reviewing PRs is real work and a big part of the team's job. When you review:

- Read the code carefully. Run it locally if you can.
- Be specific. "This is confusing" is less useful than "I'd rename `x` to `coherence_block_size` because it's not obvious what `x` represents here."
- Ask questions when you don't understand. The goal is shared understanding, not gatekeeping.
- Approve when it's good enough, not when it's perfect. Perfection is the enemy of progress on a class project.

### Merging

Once approved, the PR author merges. Use **Squash and merge** for most PRs — it collapses your many small commits into one tidy commit on `main`, which keeps the history clean.

After merging, delete your branch (GitHub offers a button right after the merge). You don't need it anymore.

---

## 7. Conventions for this repo

These are the rules we agree to follow.

### Repo layout (planned)

```
csi_sensing/
├── README.md                     # Top-level entry point
├── docs/                         # All explanation docs
│   ├── 01-foundations.md
│   ├── 02-csi-and-ssl.md
│   ├── 03-the-project.md
│   ├── 04-github-workflow.md
│   └── 05-first-week-checklist.md
├── src/                          # All Python code
│   ├── data/                     # Dataset loaders, preprocessing
│   ├── augmentations/            # Each of our four augmentations
│   ├── models/                   # Encoder backbones
│   ├── ssl/                      # SimCLR, Barlow Twins implementations
│   ├── training/                 # Pre-training and fine-tuning loops
│   └── evaluation/               # Cross-domain protocols, metrics
├── configs/                      # YAML config files for experiments
├── scripts/                      # One-off scripts: download data, run experiment
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit tests
├── results/                      # Saved experiment outputs (gitignored if heavy)
└── requirements.txt              # Python dependencies
```

### Code style

- Python 3.10+.
- We use **PyTorch** as the deep-learning framework.
- Use `black` for formatting and `ruff` for linting. We'll add a **CI** check that enforces this. ("CI" is *continuous integration* — automated checks that run on every pull request to catch problems before merge.)
- Type hints on function signatures wherever practical.
- Module docstrings explaining what each file does.

Files that shouldn't be tracked in the repo (large datasets, trained model weights, your local virtual environment) are listed in a `.gitignore` file at the repo root. Git automatically ignores anything matched by patterns in this file. If you find yourself wanting to commit a multi-gigabyte file, that's a sign it should be added to `.gitignore` instead.

### Experiment logs

Every experiment produces a directory under `results/<date>-<short-description>/` containing:
- The exact config that was run.
- The git commit hash at run time.
- All output: training curves, final metrics, model checkpoints (if small enough).
- A short `notes.md` written by the person who ran it: what they expected, what they saw, anything that surprised them.

This is non-negotiable. The whole point of the project is producing knowledge, and undocumented experiments produce no knowledge.

### Reproducibility

We use **PyTorch** as our deep-learning framework. Set seeds. Always. At the top of every training script:

```python
import random, numpy as np, torch
SEED = 42  # or whatever; pick from a small set we agree on
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

Document the seed in your experiment log. We always run at least 3 seeds before claiming a result.

---

## 8. Communication norms

The project moves at the speed of communication, not the speed of typing. A few rules:

- **Ask questions early.** If you've been stuck on something for more than an hour, ask. Tag the issue with `help-wanted` and post in the team chat.
- **Status updates are short and concrete.** "Done with the Widar3.0 loader, validated on 100 samples, PR up at #15." Not "working on stuff."
- **Disagreement is fine, even good.** If you think the plan is wrong, say so — in the issue thread, with a specific argument. The plan is a hypothesis; we update it when we learn things.
- **Don't merge your own PR if there's any doubt.** When in doubt, ask another teammate to look at it.

---

## 9. The first thing to do as a new teammate

If you have just been added to the repo, the next document — [`05-first-week-checklist.md`](05-first-week-checklist.md) — walks you through your first hour, your first pull request, and the errors you'll most likely hit. Read that next.

The short version: clone the repo, pick a `good-first-issue`, open a branch, commit small, open a PR, address feedback, merge, delete the branch. The first PR is always the slowest. The second is faster. By the third you'll be moving at full speed.

---

## 10. If something goes wrong

You broke `main`? It happens. Tell someone immediately. We revert the merge or apply a hotfix. Nobody is angry — broken `main` is everyone's problem to fix, not just yours.

You committed something secret (a password, a private key)? Tell someone immediately. Don't try to "just delete the commit" — git history is hard to truly erase, and we may need to rotate the secret.

Your branch is hopelessly behind `main`? Either rebase (an advanced git operation that *replays* your commits on top of the latest `main` so they appear as if you'd worked from the new base) or just close the PR, branch off fresh `main`, and reapply your changes manually. Sometimes starting over is faster than untangling.

You hit a **merge conflict**? This happens when git can't automatically combine your changes with someone else's because you both edited the same lines. Don't panic. Git marks the conflicting region in the file with `<<<<<<<`, `=======`, and `>>>>>>>` markers around the two competing versions. Open the file, decide which version (or which combination) you want, delete the markers, then `git add` the file and `git commit` to finish the merge. If you're unsure, ask — fixing a conflict together is a 10-minute job.

You don't understand what code on `main` is doing? Open an issue with `area:docs`. The fact that you don't understand it is the bug.
