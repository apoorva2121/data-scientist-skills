---
name: ds-method-comparison
description: Use when deciding which of several preprocessing methods, feature sets or models is best for a tabular or time-series problem; when running hyperparameter tuning or a pipeline search (Optuna, grid, random, halving, AutoML); when two options score close together and it is unclear whether the difference is real; when combining the best choices from several pipeline stages; or when building an ensemble.
---

# Method Comparison

## Overview

Two modes, one discipline.

- **Screening (Stages 4-7).** Inside one stage, compare that stage's candidates while everything
  else stays fixed, using two proxy models. Keep the best two or three, not one.
- **Joint search (Stage 8).** Treat every kept option as a categorical hyperparameter next to
  the model's own, in a conditional space, and search the combinations.

The discipline in both: every candidate is scored on **the same folds**, compared by **paired,
overlap-corrected differences**, and anything within the noise is a **tie that goes to the simpler
option**. REQUIRED BACKGROUND: `ml-experiment-measurement` (noise floor, marginal gain).

Why two phases: a choice that is best in isolation is not always best in combination. Native
missing-value handling only exists in boosted trees; scaling only matters to linear and distance
models; an outlier cap that helps a linear model does nothing for a tree. Screening narrows the
space cheaply; the joint search finds the combination.

## Mode 1: Screening within a stage

**Proxies.** Fixed hyperparameters during screening, so differences come from the stage's choice:

| Proxy | Classification | Regression | Default preprocessing for other slots |
|---|---|---|---|
| linear | `LogisticRegression(C=1, max_iter=3000)` | `Ridge(alpha=1)` | median + indicator, one-hot, `StandardScaler` |
| gbdt | LightGBM, 300 trees, lr 0.05, 15 leaves, 30 min child | same, regression objective | native NaN, one-hot (or native categorical) |

If LightGBM is not installed, `HistGradientBoostingClassifier/Regressor` is the gbdt proxy.
Unsupervised problems use the stage's own quality measure instead (stability, silhouette).

**One change at a time.** Vary only this stage's slot. Other slots stay at the reference: the best
carried-forward option from earlier stages, or the simple default above.

**Folds.** Use `load_split` from `ds-validation-design`. For independent-row schemes, screen with
repeated CV on the train part (at least 10 folds in total, for example 5x2), because five paired
differences give a noisy standard error. For grouped and time schemes the folds are fixed: use them
with 3 model seeds; seeds are averaged within each fold before pairing.

```python
import sys; sys.path.insert(0, "<path to ds-method-comparison/scripts>")
from screen import evaluate, summarize, topk_scorer
long = evaluate({"linear|median": p1, "linear|knn": p2, "gbdt|native": p3}, X, y, folds=folds,
                scoring="average_precision", seeds=range(3), seed_param="model__random_state")
for fam in ("linear", "gbdt"):
    print(summarize(long[long.name.str.startswith(fam)], noise_floor_sd=NOISE))
```

`summarize` marks `keep=True` when a candidate is not worse than the best by more than one
corrected standard error of the **paired** difference. The correction (Nadeau and Bengio) accounts
for overlapping CV training sets. Pairing on shared folds cancels fold difficulty, so this is far
more sensitive than comparing means against the Stage 2 noise floor. Use the noise floor for numbers
not measured on the same folds (CV mean against holdout, this run against an earlier one); it is
shown as `within_noise_floor` for reference and does not drive `keep`.

**What to carry forward** (write to `search_space.json`):
- every `keep=True` candidate, up to three per family, simplest listed first;
- the best candidate for each family, even when another family's option is better overall, so the
  joint search can still build a complete pipeline for every family;
- nothing else. Every dropped candidate gets a reason in the decision log. (the orchestrator's carry-forward paragraph lists the two exceptions: an unserved family, and a cheap family marginally behind at defaults)

**Cost control.** Time one fold per candidate before running all folds. A candidate that is
expensive on the full data (KNN or iterative imputation on a million rows) is screened on a
stratified sample, and the log says so.

## Mode 2: Joint search (Stage 8)

**Prerequisites:** `search_space.json` with all stages filled, the noise floor, and the model
families that survived Stage 7.

**Adapt the template** `scripts/joint_search.py`. Only its ADAPT sections change: column lists, slot
builders (every builder returns a transformer fitted in-fold, never a precomputed column), and model
families with their search ranges. List each slot's options simplest first; that order breaks ties.

```bash
python scripts/joint_search.py --data cleaned.csv --split split/ --space search_space.json \
    --id <id> --target <t> --trials 60 --timeout 900 --top 5 --seeds 3 --out search/
```

What it does: TPE (multivariate, grouped, for the conditional space) with median pruning across
folds; then the top five **distinct pipelines** (family plus slot choices, each at its best-tuned
trial) are re-scored with three seeds; the one-standard-error rule picks the simplest pipeline
within noise of the best; the pick and the best-by-mean are refitted on the train part and scored
on the locked holdout. On a few thousand rows, 40-60 trials of linear and boosted families take a
few minutes on a laptop CPU; time one trial before setting the budget.

**Budget.** Time one trial, then set `--trials` to what the budget allows. As a floor, allow 15
start-up trials plus about 10 per categorical dimension. If the right search does not fit the
budget, shrink the carried-forward sets or ask the user for more time or compute; do not quietly
run a search too small to find anything.

**Reading the holdout.**
- Pick and best-by-mean within noise of each other on the holdout: keep the pick.
- Pick's holdout score worse than its CV mean by more than about two standard errors of the holdout
  estimate: the search overfit its folds, or the holdout period drifted. Prefer the simpler finalist,
  and check drift with adversarial validation before going further.

**Final refit.** Refit the chosen pipeline on train plus holdout (everything except test) with the
chosen hyperparameters. For boosted models tuned with a fixed number of trees, keep that number;
if early stopping was used, fix the tree count found in CV rather than early-stopping on the test set.

### Early stopping leaks into the score it is measured by

The common failure is not early-stopping on the *test* set. It is early-stopping on the **fold you
are about to score**, which is the default in almost every boosting tutorial:

```python
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(200)])   # then score X_va
```

The stopping round was chosen by watching `X_va`, so the reported fold score is optimistic. The
bias is small per fold and perfectly consistent across them, which makes it invisible to a paired
test and to any amount of repeated CV - every candidate is inflated, so nothing looks wrong.

**Honest early stopping:** hold out a slice of the *training* portion of the fold, choose the round
count on that slice, then refit on the full training portion for that round count, and only then
score the validation fold.

```python
Xa, Xb, ya, yb = train_test_split(X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=0)
n = fit_with_early_stopping(Xa, ya, eval_set=(Xb, yb)).best_iteration_
final = Model(n_estimators=int(n * 1.11)).fit(X_tr, y_tr)     # x1.11 restores the withheld 10%
score(final, X_va, y_va)
```

Measure the bias once rather than assuming it: run both and report the difference. And whenever a
**shuffled-label canary** is run, early stopping must be **off** - otherwise stopping adapts to the
permuted labels and the canary reads above chance for a pipeline that is perfectly clean.

### Validate the search space before spending compute on it

`joint_search.py` only offers an option when its name appears in `SLOTS[slot]` **and** the family
appears in that option's `applies_to`. Both are silent filters: a `search_space.json` entry named
after a Stage 4 candidate rather than a slot option, or an `applies_to` list written before a
family was added at Stage 7, produces **zero options for that slot** and the slot quietly defaults.
The search then reports a clean result over a space much smaller than the one you designed.

Run `joint_search.py ... --validate-only` before the real search and read the printed effective
space. It warns on unmatched option names, on any slot x family pair with zero or exactly one
option, and on any family defined in `FAMILIES` that no `applies_to` list mentions. A slot with one
option is not being searched; a slot with zero is not even being defaulted deliberately.

## Ensembles (after the pick, optional)

Average or stack the out-of-fold predictions of the distinct finalists. Accept the ensemble only if
its gain over the single pick exceeds the noise floor on the same folds and holds on the holdout.
Measure each member by its **marginal** gain (ensemble with it minus without it). A stacked
meta-learner is fitted on out-of-fold predictions only, with a simple linear model.

## Other search tools

`references/search-methods.md` covers grid, random, halving, Hyperband, Gaussian-process and
CMA-ES samplers, multi-objective search, and AutoML frameworks, with versions checked 2026-09-27.
An AutoML run is a useful **ceiling check**: if it beats the pick by more than the noise floor under
the same folds, find out what it did differently before accepting either.

## Gate (all must exist)

- screening tables per stage and the updated `search_space.json` (Stages 4-7)
- `search/trials.csv`, `search/finalists.csv`, `search/result.json` (Stage 8)
- decision-log block: pick, runners-up within 1 SE, why the pick was preferred, holdout reading
- the refitted pipeline, fitted on train plus holdout, saved for Stage 9

## Red flags

| Thought | Reality |
|---|---|
| "B beats A by 0.004 on mean CV" | Look at the paired corrected SE and the noise floor first |
| "Five folds is plenty to compare" | Five paired differences give a noisy SE; repeat the CV |
| "Let the search decide whether to drop the leaky column" | It will keep it. Correctness is fixed before the search |
| "Top five trials are all the same pipeline, fine" | Finalists must be distinct designs or the 1-SE rule has nothing to choose |
| "Holdout disagrees, so trust CV" | A holdout drop beyond noise means overfit search or drift; investigate |
| "Tune on the test set, just the threshold" | Thresholds are chosen on validation predictions at Stage 9 |
| "The ensemble adds 0.002" | Below the noise floor it is complexity, not gain |
