# 05 — Your First Week

This document is for the moment you've been added to the repo and you're staring at it thinking "okay, now what?". By the end you should have made one pull request, reviewed someone else's, and seen your name on `main`. That's the goal of week one. Real research work starts in week two.

If you haven't read [`04-github-workflow.md`](04-github-workflow.md), read it first. This doc assumes you know what a branch and a pull request are, even if you've never made one.

---

## 1. The first hour

Before you can do anything else, you need a working setup. Roughly thirty to sixty minutes, mostly waiting for downloads.

**Install the tools.**

- **git** — [https://git-scm.com/downloads](https://git-scm.com/downloads). On Windows this also gives you "Git Bash," which is the easiest place to run the commands in this doc.
- **Python 3.10 or newer** — [https://www.python.org/downloads/](https://www.python.org/downloads/). On the installer screen, tick the "Add Python to PATH" box. If you forget, you can re-run the installer.
- **A code editor** — VS Code ([https://code.visualstudio.com/](https://code.visualstudio.com/)) is what most of the team uses. Anything works.

**Make a GitHub account** at [https://github.com](https://github.com) if you don't have one. Send your username to George so he can add you to the repo.

**Clone the repo.** "Cloning" means downloading a copy to your laptop. Open a terminal (Git Bash on Windows, Terminal on macOS, your shell on Linux) and run:

```bash
git clone https://github.com/Jupiter-Plantagenet/csi_sensing.git
cd csi_sensing
```

If you get an error that says "authentication failed" or "repository not found," it means GitHub doesn't recognize you yet — message George.

**Tell git who you are.** Commits are stamped with your name and email. Set them once, ever:

```bash
git config user.name  "Your Name"
git config user.email "your-email@example.com"
```

Use the same email as your GitHub account. Otherwise GitHub won't link your commits to your profile picture.

**Sanity check.** Run `git status`. You should see something like `On branch main` and `nothing to commit, working tree clean`. If you do, your setup is working.

---

## 2. Your first pull request

Your first issue will be **"add yourself to `CONTRIBUTORS.md`."** Every teammate gets one. It is deliberately tiny: the goal is not the contribution, it is the practice.

Here is the whole loop, step by step.

**Step 1 — make sure your `main` is up to date.** Other people may have merged things since you cloned. Always do this before starting a new branch.

```bash
git checkout main
git pull
```

**Step 2 — make a branch for your change.** Name it `your-name/add-self-to-contributors`. Keep your name short and use lowercase.

```bash
git checkout -b george/add-self-to-contributors
```

The `-b` means "create the branch and switch to it." If you forget the `-b`, git will complain that the branch doesn't exist.

**Step 3 — make the change.** Open `CONTRIBUTORS.md` in your editor. Add one line under the existing list:

```
- Your Name (your student ID) — one sentence about what you want to learn from this project.
```

Save the file.

**Step 4 — see what you changed.** This habit will save you many times over.

```bash
git status      # which files changed
git diff        # what exactly changed inside them
```

If `git diff` shows what you expected, you're ready.

**Step 5 — stage and commit.** "Staging" tells git which changes to include in the next commit; "committing" saves them as a snapshot with a message.

```bash
git add CONTRIBUTORS.md
git commit -m "Add <your name> to contributors"
```

Use the imperative ("Add" not "Added"). Keep it under 72 characters. See [`04-github-workflow.md`](04-github-workflow.md#5-doing-the-work) for the full commit-message rules.

**Step 6 — push your branch to GitHub.**

```bash
git push -u origin george/add-self-to-contributors
```

The `-u` part tells git to remember the connection between your local branch and the remote one. From now on, on this branch, plain `git push` is enough.

**Step 7 — open the pull request.** Go to the repo on GitHub. There will be a yellow banner offering a "Compare & pull request" button. Click it. The PR template will already be filled in with section headers — replace each placeholder with one or two sentences. For this PR they'll be very short:

- *What this does:* adds my name to the contributors list.
- *Why:* closes #N (the issue number you were assigned).
- *How I tested it:* opened the file locally, confirmed the line renders correctly in GitHub's preview.
- *Anything reviewers should pay attention to:* nothing.

Click "Create pull request."

**Step 8 — request a review.** On the right side of the PR page, click "Reviewers" and pick a teammate. Post in the team chat that your PR is up.

**Step 9 — wait for approval, then merge.** Once a reviewer clicks "Approve," you'll see a green "Squash and merge" button on the PR page. Click it. Confirm. GitHub will then offer to delete the branch — click that too. You don't need it anymore.

**Step 10 — pull the new `main` to your laptop** so your local copy includes your own merge:

```bash
git checkout main
git pull
```

That's it. You've done the whole loop. Every future PR follows the same ten steps.

---

## 3. Reviewing someone else's pull request

Reviewing is half the team's work. The first time someone asks you to review, do this:

1. Open the PR on GitHub. Read the description.
2. Click the "Files changed" tab. Read every changed line.
3. If something is unclear, leave a comment on that specific line — click the `+` icon that appears when you hover. "What does this variable represent?" is a good comment. "This is wrong" without explanation is not.
4. When you're satisfied, click "Review changes" at the top right and pick **Approve**, **Request changes**, or **Comment**:
   - **Approve** — looks good, the author can merge.
   - **Request changes** — there's something the author needs to fix before merging.
   - **Comment** — you have thoughts but they're not blocking.

For the "add yourself to contributors" PRs, approval is almost always the right answer. The point is the practice.

See [`04-github-workflow.md`](04-github-workflow.md#reviewing-someone-elses-pr) for more detailed review guidance once the PRs get bigger.

---

## 4. Common errors and what they mean

These will happen. They are not signs that something is broken — they are signs that you're using git the way everyone uses git.

**`fatal: not a git repository`** — you're running a git command outside the repo folder. Run `cd csi_sensing` (or wherever you cloned it) first.

**`Authentication failed`** when you push or pull. GitHub doesn't recognize you. Ask George to add you to the org, or set up an [SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) (one-time setup, never deal with passwords again).

**`Your branch is behind 'origin/main' by N commits`** — other people merged things while you were working. Run `git pull` to catch up.

**`CONFLICT (content): Merge conflict in some/file.py`** — you and someone else both edited the same lines. Open the file. You'll see markers like:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```

Decide which version (or which mix) you want. Delete the markers and the version you don't want. Save. Then:

```bash
git add some/file.py
git commit
```

If you're unsure, ask. Resolving a conflict together takes ten minutes. Resolving it wrong silently loses work.

**`error: failed to push some refs`** — usually means the remote has commits your local branch doesn't. Run `git pull`, fix any conflicts, then push again.

**You committed something secret** (a password, an API key). Don't try to "just delete the commit." Tell the team immediately. The secret has to be rotated, because git history is hard to truly erase.

**You can't figure out what's going on.** That is fine. Ask. The fastest way to learn git is to watch someone else use it for ten minutes.

---

## 5. Where to go next

Once your first PR is merged:

1. Look at the **Issues** tab. Find an issue labeled `good-first-issue` that isn't assigned. Comment "I'll take this" and assign yourself.
2. Repeat the ten steps in section 2, but with real work this time.
3. Read the open *design discussion* issues — the ones that come from [`03-the-project.md`](03-the-project.md) section 7. You don't have to write code to contribute; a thoughtful comment is also a contribution.

By your third PR, this will all feel routine.

---

Continue (when you're ready) back to [`03-the-project.md`](03-the-project.md) for the research plan, or [`04-github-workflow.md`](04-github-workflow.md) for the full workflow rules.
