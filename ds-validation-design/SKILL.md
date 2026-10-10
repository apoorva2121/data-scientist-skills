---
name: ds-validation-design
description: Use when choosing how to split data for model validation - train/test split, cross-validation, K-fold, stratified, grouped, time-based or rolling-origin backtesting, nested CV - before any model is compared; when rows repeat per customer, patient, device, store or property; when data is time-ordered or drifting; when a model validated well but failed on new data; or when deciding whether a metric difference between two models is real.
---

# Validation Design

## Overview

The split must mirror how the model will be used. A random split answers "how well does it
predict rows like these, from entities it has seen, in the same period". If deployment means new
customers, or next quarter, that is the wrong question and the score will be optimistic.

This stage produces three things, once, before any modelling: a frozen **test set**, a locked
**holdout** for the Stage 8 search, and **CV folds** that every later stage screens on. It also
measures the **noise floor**, the metric wobble that later differences must beat.

## 1. Answer the deployment question

Write the answers in the decision log; they choose the scheme.

- At prediction time, is the model scoring **new rows of known entities**, **entirely new
  entities**, or **a future period**? Often more than one.
- Does the relationship between features and target **drift** over time? (Stage 1 target rate per
  period is the evidence.)
- Do labels or features **overlap in time** between rows (rolling windows, multi-day outcomes)?

| Data structure and deployment | Scheme (`make_split.py --scheme`) | Why |
|---|---|---|
| Independent rows, no drift, classification | `stratified` | Keeps the class ratio in every part |
| Independent rows, no drift, regression | `stratified` (quantile bins) or `random` | Bins keep the target range in every fold |
| Entities repeat; deployment on new entities | `group` | The same entity on both sides inflates the score |
| Drift, or deployment on a future period | `time` | Tests extrapolation to the future |
| Entities repeat and time matters | `group_time`, then decide on overlap (below) | Time order first, entity policy second |
| Forecasting (target is a future value of a series) | `group_time` with `--test-periods H`, `--holdout-periods H` | Last H periods of every series; expanding-window folds |
| Features built from look-back windows | add `--gap-periods` = window overlap | Stops the validation block leaking into training features |
| Labels overlap in time (finance, multi-day outcomes) | `time` + purge + embargo (below) | Overlapping labels leak across the boundary |
| Spatial data | `group` on a spatial block (grid cell, region) | Neighbours are correlated |
| Under about 1,000 rows | `stratified` with repeats + nested CV (below) | A single split is too noisy to choose anything |

**Entity overlap under `group_time`:** if the entity's earlier history exists at prediction time (a
known property resold, an existing customer), overlap is realistic and time order already prevents
leakage. If deployment is on new entities, drop test-period entities from training with
`make_split.py --drop-overlap-groups` (their rows become `part = excluded`) and say so.

**Purged K-fold with embargo:** remove from training any row whose label window overlaps the
validation block (purge), plus a buffer after it (embargo). Implement by filtering the fold index
pairs from `cv_folds.py` on each row's label start and end times.

## 2. Carve and freeze

```bash
python scripts/make_split.py cleaned.csv --target <t> --scheme <scheme> --id <id> \
    [--group <g>] [--time <date>] [--test-size 0.2 | --test-periods H] \
    [--holdout-size 0.15 | --holdout-periods H] [--n-splits 5] [--gap-periods G] --seed 42 --out split/
```

It writes `split/split.csv` (id, part, fold, period) and `split/split_spec.json` (sizes, target per
part, time ranges, group overlaps, and a SHA-256 of the test IDs). Load folds in later stages with:

```python
import sys; sys.path.insert(0, "<path to ds-validation-design/scripts>")
from cv_folds import load_split, verify_test_untouched
s = load_split("split/split.csv", ids=df["<id>"])
train = df[s.train_mask]          # all screening and CV happens here
cross_val_score(pipe, X_train, y_train, cv=s.folds, scoring=...)
```

The three parts have fixed jobs:

| Part | Used at | Job |
|---|---|---|
| train (with folds) | Stages 3-8 | EDA, screening, carry-forward decisions, the joint search |
| holdout | Stage 8 only | Confirms the search winner; the search overfits its own folds |
| test | Stage 9 only, once | Reports the final number. Chooses nothing |

Before Stage 9 scores the test set, `verify_test_untouched("split")` must return True.

## 3. Is the test set big enough?

Estimate the 95% interval the test set can deliver. If it is wider than the smallest difference
that matters to the business, enlarge the test set or report nested-CV estimates alongside it.

| Metric | Standard error (approximate) |
|---|---|
| Proportion (accuracy, precision@k, recall) | sqrt(p(1-p)/n), with n = the rows the proportion is taken over |
| ROC-AUC | Hanley-McNeil: needs positives n1, negatives n0 |
| MAE, RMSE | sd of the per-row errors / sqrt(n) |

Worked example (Hanley-McNeil): 150 positives and 1,850 negatives at AUC 0.75 gives a standard
error of about 0.024, so a 95% interval of roughly plus or minus 0.05. Two models 0.01 apart are
indistinguishable on that test set, which is why choices are made on repeated CV, not the test set.

## 4. Small and large data

- **Small data** (roughly under 1,000 rows or under 100 positives): use repeated stratified CV
  (5x5 or 10x3). Tune inside an inner loop and estimate performance in the outer loop (nested CV);
  a single holdout cannot both tune and estimate honestly at this size. Consider a smaller
  holdout, or none, and state that the nested-CV estimate replaces it.
- **Large data** (millions of rows): one validation split is stable enough; screen on a stratified
  sample and confirm survivors on the full training set.

## 5. Noise floor

Measure it now, with one simple pipeline, so every later comparison has a yardstick.
REQUIRED BACKGROUND: `ml-experiment-measurement`.

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)   # iid schemes
scores = cross_val_score(simple_pipe, X_train, y_train, cv=cv, scoring=metric)
fold_sd = scores.std(ddof=1)
repeat_sd = scores.reshape(5, 5).mean(axis=1).std(ddof=1)
```

Measure it **in the primary metric's own units**: use the Stage 0 primary metric as `scoring`
(for a capacity metric, `topk_scorer(0.10)` from `ds-method-comparison/scripts/screen.py`); a
noise floor measured in ROC-AUC says nothing about precision at 10%.

**`repeat_sd` does not measure sampling noise at any n.** Repeats reshuffle *partitions of the
same rows*; they never resample data. Checked against a row-resampling estimate on synthetic data
it came out 2x too small at n=1,000, 7x at n=5,000 and 58x at n=30,000, and on one real
dataset of several hundred thousand rows it was three hundred times too small. Adopting it would have
called a 0.0001 difference significant. There is no n at which it is safe by default and no clean
threshold for when it has "degenerated"; it answers a different question.

**Default: estimate the noise floor by resampling the rows, not the partition.** Draw several
subsamples **without replacement** (with replacement puts duplicate rows on both sides of a fold
and inflates the level for tree and target-encoded pipelines), run CV inside each, and take the sd
of the CV means:

```python
rng = np.random.default_rng(0); means = []
n = len(X_train); m = int(0.8 * n)
for b in range(8):
    idx = rng.choice(n, m, replace=False)
    s = cross_val_score(simple_pipe, X_train.iloc[idx], y_train.iloc[idx],
                        cv=StratifiedKFold(5, shuffle=True, random_state=b), scoring=metric)
    means.append(s.mean())
noise_floor_sd = float(np.std(means, ddof=1))
```

Record `repeat_sd` alongside for reference, never as the floor. The two answer different
questions and disagree by orders of magnitude: the paired SE (and repeat_sd) ask "is this
difference consistent on *these* rows"; the subsample sd asks "would it replicate on *other*
rows". Carry-forward decisions need the second.

For reference only, `repeat_sd` is the spread of the CV *mean*,
because later comparisons are between means. `fold_sd` is the spread of single folds and is
roughly the square root of the number of folds times larger; using it as the noise floor makes
almost every candidate look tied. For grouped and time schemes the folds are fixed: rerun with 5
model seeds on `s.folds` and use the sd of the CV mean across seeds; if the model is deterministic,
use `fold_sd / sqrt(number of folds)`. Later, compare two pipelines by their **paired per-fold
differences** on the same folds, not by their means alone.

## 5b. Replicate on a second split before carrying anything forward

One partition is one draw. On shared folds the paired standard error is small by construction, so
"consistent across folds" is nearly automatic and is **not** evidence that a difference will
survive a different partition of the same rows.

**Rule: any choice carried into Stage 8 must hold on two independently seeded fold partitions of
the training portion.**

**Split B is built INSIDE the training portion. It never re-carves the test set.** Do not run
`make_split.py` a second time with a different seed: that draws a different test set from all
rows, and about two thirds of split A's frozen test rows land in split B's training part -
screening on B then trains on test rows and violates non-negotiable #2. Reshuffle only the folds
over `s.train_mask`:

```python
from sklearn.model_selection import StratifiedKFold
s = load_split("split/split.csv", ids=df["<id>"])          # split A: test and holdout stay fixed
X_tr, y_tr = X[s.train_mask], y[s.train_mask]
folds_B = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X_tr, y_tr))
# folds_A = s.folds ; both index the SAME training rows, differently partitioned
```

For grouped or time schemes, reshuffle with the matching splitter (`GroupKFold` after a seeded
permutation of the group labels, or a different origin offset) - again over the training portion
only.

Screen on folds A, then re-screen the survivors on folds B. Accept only what clears the bar on
**both**, and record both numbers. A candidate that wins on A and loses on B is noise that happened
to land well, which is exactly the failure a single partition cannot show you.

Cost: one extra screening pass over the survivors only, not over the whole menu. On large data
this is minutes. It is the cheapest real guard in the whole process, and the one most often
skipped because the first partition already produced a satisfying answer.

## 5c. What the test carve-out costs, and how to report it

Carving 20% test plus 15% holdout removes **35% of the data from model selection**. That is the
price of an honest, once-scored number, and non-negotiable #2 says it is paid: the test set is
carved at Stage 2 and scored once at Stage 9. What this section adds is that the price is
**measurable and must be reported**, not that it can be avoided.

| Situation | Cost of the carve-out | What to do |
|---|---|---|
| Learning curve has plateaued | Near zero | Report it as such |
| Learning curve still rising at full size | Real: selection happens on a weaker model than the one shipped | **Report the cost** next to the test score (below) |
| The delivered model is refit on 100% of the data (competitions, batch scoring) | Selection and delivery disagree: hyperparameters tuned on 65% are applied at 100% | Report it; if re-tuning tree count or regularisation on the full refit, fix the rule in advance and never look at test |
| Under ~1,000 rows | Severe | The small-data rule in section 4 applies (nested CV for the *estimate*); it still never tunes on test rows |

**Measure the cost, do not guess it.** After Stage 7's learning curve exists, read the validation
score at the training-portion size and extrapolate to the full-data size; the difference is what
the carve-out costs, and it belongs in the report beside the test score. A run that reports
"test AUC 0.9455" without noting that the same pipeline would plausibly score higher had selection
seen all the data is telling half the story. Treat the extrapolated figure as **plausible, not
measured** unless it exceeds the noise floor.

## 6. Sanity checks on the split

- Target distribution per part (in `split_spec.json`) matches, except where drift is the point.
- `group_overlap` is 0 wherever the policy says it must be; `folds_sharing_groups` is 0.
- Time ranges do not overlap between train, holdout and test for time schemes.
- Adversarial validation, train against test (`ds-data-audit`): near 0.5 for random schemes. A
  high AUC there means the split or the data is broken. For time schemes a high AUC is expected;
  it measures the drift the model must survive.

## Unsupervised path: stability instead of a test set

Fit the method on many subsamples (for example 20 draws of 80%). Compare assignments on the shared
rows with the adjusted Rand index. Mean ARI near 1 means stable structure; below about 0.6, the
clusters are artefacts of the sample. When a few anomaly labels exist, hold them out for evaluation only.

## Gate (all must exist)

- deployment answers and the chosen scheme, with at least one rejected scheme and why, in the decision log
- `split/split.csv` and `split/split_spec.json`, with sanity checks passed
- test-set precision estimate for the primary metric
- noise floor recorded in `search_space.json` as the sd of the CV mean over **row subsamples**
  (section 5), with `repeat_sd` and `fold_sd` alongside for reference only
- a second fold partition of the **training portion** exists (`folds_B`, section 5b), and the gate
  states that survivors will be re-screened on it. It does not re-carve the test set
- the carve-out cost is named: what fraction of the data selection will not see (section 5c)

## Red flags

| Thought | Reality |
|---|---|
| "Rows look independent, random K-fold" | Check the `GROUP_KEY?` and date columns from Stage 1 first |
| "TimeSeriesSplit is enough" | Without a gap, look-back features leak the validation block into training |
| "Test AUC went up 0.01, ship it" | With 150 positives the test interval is roughly plus or minus 0.05 |
| "Holdout is wasted data" | The search overfits its own folds; the holdout is how you find out |
| "It held on all 5 folds, that's solid" | Same rows, one partition. Re-screen survivors on a second seed |
| "repeat_sd is tiny, so everything is significant" | It never measured sampling noise. Subsample the rows instead |
| "Second split = make_split.py with another seed" | That re-carves the test set. Reshuffle folds inside the training portion only |
| "The carve-out is just standard practice" | It removes 35% from selection. Price it against the learning curve and report it |
| "Tune on the test set just this once" | Then it is a validation set and you have no test set |
| "Split after feature engineering" | Features fitted on all rows have already seen the test set |
