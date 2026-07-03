# Pre-Registration: Entropy-Trajectory Reasoning-Reliability Reproduction and Reasoning-Model Stress-Test

- **Version:** draft v0.1 (2026-07-03) — DRAFT, not yet registered. Sign-off items in §12.
- **Authors:** T. Cochran (A4A); co-authorship to be decided after a draft.
- **Registry:** Open Science Framework (OSF), Standard Pre-Registration. Register **before
  any model runs**; freeze the git commit hash of this file in the OSF entry.
- **Study type:** computational reproduction plus a pre-specified extension. Confirmatory
  and exploratory analyses are separated explicitly (§4 vs §10).

> [!important] Registration discipline
> All parameters below are fixed from Zhao (2026)'s published protocol, not tuned on our
> own data, which is what licenses registering before any runs. The mock-backend smoke
> test uses no model and is not a run in this sense. Any post-registration departure is
> reported as a labeled deviation (§11).

## 1. Background and the finding under test

Zhao (2026) reports that for chain-of-thought answers, the **shape** of the per-step answer
entropy trajectory (whether it decreases monotonically) predicts correctness, while the
**magnitude** of total entropy reduction does not. We (a) independently reproduce this on
open-weight models, (b) stress-test it on a reasoning-distilled model the original did not
cover, and (c) test whether a small learned filter reduces the monotonicity rule's
false-positive rate.

## 2. Design summary

For each problem: sample a reference chain, segment it into cumulative prefixes, sample
`m = 5` short continuations at each prefix (temperature 0.7, <=150 tokens), extract the
answer from each, compute the Shannon-entropy trajectory, and derive: binary
epsilon-monotonicity (epsilon = 0.01), violation count, coherence magnitude
(C = H_0 - H_N), and final-answer confidence (last-step self-consistency). Correctness is
the majority final answer vs the gold answer.

## 3. Models, datasets, sample sizes

- **Model panel** (exact Hub revision SHAs to be pinned at registration, §12):
  Qwen/Qwen2.5-7B-Instruct (**anchor**), mistralai/Mistral-7B-Instruct-v0.3,
  meta-llama/Llama-3.1-8B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (reasoning-distilled).
- **Datasets:** GSM8K test (full, n = 1319) and MATH-500 (n = 500). Full test sets are
  used; there is no data-dependent or optional stopping.
- **Seeds:** one primary seed per (model, dataset) cell; a 3-seed robustness check is a
  pre-specified secondary analysis (§10).

## 4. Confirmatory hypotheses and analyses

Primary inference is by **95% bootstrap confidence intervals** (1000 resamples over
problems); p-values are secondary. The confirmatory family is H1-H3 and H5; multiplicity
across the primary family is controlled with **Holm-Bonferroni**.

- **H1 (replication, primary; directional).** For the anchor model, monotone chains have
  higher accuracy than non-monotone chains on both GSM8K and MATH-500. Test: 2x2
  (monotone x correct), accuracy gap with 95% CI, Fisher exact (one-sided, greater), odds
  ratio. **Success criterion:** gap CI excludes 0 with positive direction on both datasets.
- **H2 (magnitude null; directional-null).** Coherence magnitude C is at most negligibly
  associated with correctness. Test: Spearman rho(C, correct) with 95% CI. **Criterion:**
  |rho| < 0.10 and substantially smaller than the shape effect.
- **H3 (graded signal; directional).** Accuracy decreases as violation count increases.
  Test: accuracy by violation bucket {0, 1, 2, >=3} and Spearman rho(violations, correct)
  with 95% CI. **Criterion:** rho < 0 with CI excluding 0.
- **H5 (false-positive reduction; directional).** A logistic filter over
  [violation-count, final-answer confidence], fit on a stratified 60% train split, achieves
  a **lower false-positive rate than the monotonicity baseline at matched coverage** on the
  held-out 40% test split. Test: coverage-matched FP-rate difference plus the full
  coverage-vs-selective-accuracy curve. **Criterion:** FP-rate reduction CI excludes 0
  (favorable direction), reported on test only.

### The reasoning-model stress-test (primary novelty; estimation, not a directional bet)

- **H4 (reasoning-distilled behavior).** We **do not** predict a direction: the shape
  signal may hold, weaken, or break on DeepSeek-R1-Distill-Qwen-7B. Pre-specified as
  estimation: report the accuracy gap and its 95% CI for the distilled model, and the
  **difference in gap** between the distilled model and the mean of the three standard
  models (95% CI). Either outcome (signal survives / signal degrades) is a publishable,
  interpretable result. This is registered as two-sided precisely so neither outcome is a
  post-hoc story.

## 5. Operationalization (fixed)

- **Entropy:** Shannon entropy over the empirical answer distribution at each prefix;
  natural log (reported units do not affect shape tests).
- **Monotonicity:** epsilon = 0.01 primary. (Robustness to epsilon in {0, 0.05, 0.10} is
  secondary, §10.)
- **Sampling:** m = 5 primary, temperature 0.7, max 150 tokens.
- **Step segmentation (the crux for the distilled model):** **primary rule = blank-line**
  segmentation of the reference chain into cumulative prefixes, capped at `max_steps = 8`.
  Alternative rules (newline, sentence, token-window at 40 tokens) are pre-specified
  **secondary sensitivity** analyses (§10), reported for all models and prominently for the
  distilled model.
- **Answer extraction:** GSM8K = last number, normalized (strip $ and thousands commas,
  drop trailing .0); MATH-500 = last \boxed{...}, normalized (strip spacing/formatting
  macros, unify \frac), else fall back to last number.
- **Correctness:** majority answer at the final prefix equals the normalized gold answer.

## 6. Sample-size rationale and stopping

Full published test sets (1319 + 500) are used in their entirety; there is no optional
stopping, no interim peeking that gates continuation, and no data-dependent exclusion.
Problems that yield fewer than two parseable answer-bearing steps are excluded (a fixed,
pre-specified rule already in the harness); the excluded count is reported.

## 7. Primary vs secondary outcomes

- **Primary:** H1 (anchor replication) and H4 (distilled stress-test estimation).
- **Secondary confirmatory:** H2, H3, H5, and replication of H1/H2/H3 on the two additional
  standard models.

## 8. Inference and multiplicity

Bootstrap 95% CIs are the primary inferential object. Where p-values are reported, the
primary confirmatory family (H1 both datasets, H3, H5) is corrected with Holm-Bonferroni.
H4 is estimation and is not part of the null-testing family.

## 9. What would count as a failed replication

If, on the anchor, the H1 gap CI includes 0 on either dataset, we report a failed or
partial replication with boundary conditions rather than reframing. An honest null is an
intended, publishable outcome.

## 10. Exploratory analyses (labeled non-confirmatory)

Reported as exploratory, not used to support confirmatory claims: epsilon robustness;
segmentation-rule sensitivity beyond the primary; the m = 10/20 sweep on a dev subset;
cross-model comparison of gap magnitudes; replication of the companion **step-level
calibration-degradation** finding; a logit-based entropy variant (open-weight only); and
early-exit triage (first k transitions).

## 11. Deviations policy

Any departure from this pre-registration (parameter, model revision, analysis) is listed
in a Deviations section of the paper with the reason and, where feasible, both the
pre-registered and revised results.

## 12. Sign-off items before registration (Ted)

1. Lock the exact **Hub revision SHAs** for the four models.
2. Confirm **H4 framing** stays direction-agnostic (estimation, two-sided).
3. Confirm **primary dataset(s)**: both GSM8K and MATH-500 as primary, or GSM8K primary and
   MATH-500 secondary.
4. Confirm the **multiplicity** choice (Holm-Bonferroni) and **bootstrap-CI-primary** stance.
5. Confirm **one primary seed** with a 3-seed robustness secondary (vs 3 seeds primary).
6. OSF project owner/title and whether the repo is linked public at registration.

## 13. Registration logistics

Create the OSF project, attach this file at its frozen commit hash, register under the
Standard Pre-Registration template, then begin runs. Mirrors the OSF pre-registration
pattern used for the Truthfulness Dashboards submission.
