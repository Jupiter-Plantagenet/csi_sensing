# 06 — Using AI Well in Project Discussions

Most of us will use AI assistants — ChatGPT, Claude, Gemini, Copilot, others — at some point on this project. That is fine, and expected. AI is a useful research assistant if you treat it as one. It is a bad way to make decisions if you treat it as an oracle.

This document is about the difference. It exists because every design discussion on this project — the Wave 3 issues, the open questions in `03-the-project.md` section 7, future questions we have not thought of — will involve teammates posting suggestions. We want those suggestions to be backed by something real, not by whatever an AI happened to say first.

Read this once. Then come back to it whenever you are about to post a substantive opinion on a design issue.

---

## 1. Two ways to use AI, in plain terms

**As a research assistant.** You direct it. You ask it to find and summarise credible papers. You ask it to list options you might not have thought of. You ask it to write a short script that runs a small numerical experiment. You read the output critically. You verify the citations. You run the script and look at the result. You form an opinion based on what you saw, not on what the AI said.

**As an oracle.** You ask "what should we do about X?", you get an answer, you post the answer. You did not verify anything. You do not know whether the citations are real. You do not know whether the recommendation actually applies to *our* project, *our* data, *our* compute budget.

The first is how a graduate researcher uses any tool. The second produces work that looks confident and is often wrong in subtle ways.

---

## 2. The recommended workflow

When a Wave 3 issue (or any design discussion) asks you to weigh in, follow this sequence. It will take you between thirty minutes and a few hours, depending on the question.

1. **Read the issue body and the comment beneath it.** Both. The body says what is being decided; the comment explains the intuition and proposes a small experiment.
2. **Frame your question for AI clearly.** Not "tell me about coherence bandwidth," but "list five published methods for estimating indoor coherence bandwidth from WiFi CSI data, with citations." The more constrained the question, the more useful the answer.
3. **Get a list of options, longer than you would have come up with alone.** Treat the list as starting points, not conclusions.
4. **Ask the AI for credible references for each option.** Demand citations with arXiv IDs, DOIs, or links to the actual paper. **Click through every one.** Throw out any that do not exist or do not say what the AI claimed they said. Hallucinated citations are common.
5. **Pick the one or two options that look most promising.** Briefly, why each: what does the literature say, how hard is it to implement, does it match our setup.
6. **Run a tiny experiment to back your pick.** This is the part most people skip and the part that matters most. The experiment can be:
   - A numerical simulation. Build a synthetic input, apply the method, plot what happens.
   - A check on real data. Download one or two samples from a public dataset (Widar3.0, CSI-Bench), apply the method, plot what happens.
   - A literature cross-check. Confirm that two or three independent published papers report numbers in the range your method produces.
   - AI can help you write the script. AI cannot tell you whether the result is right. You read the plot.
7. **Post your suggestion as a comment on the issue.** Include: the options you considered, why you picked the one you did, the experiment you ran, the plot or output, your conclusion. If you used AI, say so and link to your prompt where useful.

---

## 3. What a good post looks like

A good post on issue #7 (static/dynamic decomposition) might read:

> I propose a Butterworth lowpass filter at 2 Hz cutoff. Reasoning:
>
> 1. Three options I considered: Butterworth lowpass, running median, and low-rank PCA. I dropped PCA because the static structure in a single CSI sample is one-dimensional in time, so the rank-1 part is approximately the same as the temporal mean — adds complexity for nothing.
> 2. Wang et al. 2019 (arXiv:1908.xxxx) use 1.5–2 Hz for indoor human activity recognition with similar CSI sample rates. I confirmed by reading the paper directly.
> 3. I built a synthetic test: a 0.5 Hz sine (slow background) plus a 4 Hz sine (fast motion) plus white noise. Applied a Butterworth lowpass at 2 Hz. The 0.5 Hz component lands cleanly in the static output, the 4 Hz component in the dynamic output. Plot attached, code in [Colab link].
> 4. Caveat: I have not tested on real Widar3.0 CSI yet. I would file a follow-up issue for that, blocked-by this one.

A bad post looks like:

> I asked ChatGPT and it said use a Butterworth filter at 2 Hz. Sounds good to me.

The first post lets a teammate evaluate your reasoning and replicate your experiment. The second post asks the team to take a chatbot's word for it.

---

## 4. When the verification is hard

Sometimes the experiment that would back your suggestion is itself a non-trivial project. That is fine and common. In that case:

- Post your suggestion with what you have.
- Open a follow-up issue for the verification work. Title it clearly, e.g. "Verify Butterworth 2 Hz preserves activity signal on real Widar3.0 CSI."
- Mark the original issue as blocked-by the verification, or as conditionally accepted pending the verification.

The point is never to abandon verification. The point is to make the cost of verification visible, so the team can plan for it.

---

## 5. Things AI is good at on this project

- Listing candidate methods you might not have considered.
- Finding and summarising published papers, with citations you must verify.
- Writing short scripts to simulate signals, apply transforms, and plot results.
- Helping translate equations from a paper into runnable code.
- Catching obvious errors in your own writing.

## 6. Things AI is bad at on this project

- Knowing what is actually in CSI-Bench, Widar3.0, or UT-HAR. It will guess and sometimes guess wrong.
- Knowing what the team has already decided. Always cross-check against the docs.
- Producing reliable citations without verification. It will hallucinate plausible-looking but non-existent papers.
- Telling you whether a result is correct. It will tell you the result looks correct, which is different.

---

## 7. The minimum standard

If your post on a design issue does not include at least one of:

- A citation you have read and verified yourself (or AI has read, not merely glossed from web search), or
- An experiment whose code or output you can show

then it is not yet ready to post. Take another hour, run the experiment, then post. The team's time is more valuable than the speed of any one comment.
