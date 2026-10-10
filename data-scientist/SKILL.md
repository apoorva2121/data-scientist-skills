---
name: data-scientist
description: Use when given a tabular or time-series dataset (CSV, Excel, parquet, SQL table) and a business goal, and asked to build, improve, compare or evaluate a predictive model - classification, regression, forecasting, churn, pricing, demand, risk scoring, propensity, fraud or anomaly detection. Also use when a model underperforms or misses its target, when scores look too good to be true, when someone asks "which algorithm should I use" or "build me the best model", or when a previous modelling attempt skipped steps. Not for NLP, images or LLM fine-tuning.
---

# Data Scientist

## Overview

Take a dataset plus a motive to a validated model through fixed stages. Each stage ends in a
**gate**: a named artifact that must exist before the next stage starts. Each stage screens
several methods and carries the best two or three forward; the modelling stage then searches
over the combinations jointly.

Most projects that miss their target do so through omission, not a missing algorithm: a leaky
column nobody questioned, an imputer fitted before the split, a random split on time-ordered
data, no baseline, one split deciding the winner, tuning against the test set. The gates exist
to make those omissions visible.

**Violating the letter of a gate is violating its spirit.** A gate answered with "checked, fine"
instead of its artifact has not passed.

## Non-negotiables

1. **Fitted things are fitted in-fold.** Imputers, scalers, encoders, outlier thresholds,
   resamplers and feature selectors learn from data, so they live inside a pipeline fitted on the
   training fold only. Stateless row-wise fixes (types, sentinel to null, spelling, exact
   duplicates, impossible values to null) may run before the split.
2. **The test set is carved at Stage 2 and scored once, at Stage 9.** It never chooses anything.
   Model and pipeline choice use cross-validation and a separate locked holdout inside training.
3. **Every technique gets a reason both ways.** Applied because (evidence from this data), or
   skipped because (reason tied to this data, model family, size or constraint). "Not needed" is
   not a reason.
4. **Correctness is not a hyperparameter.** Leak removal, split design, sentinel handling and
   test discipline are decided once, on evidence, and never searched over. A search would pick
   the leak.
5. **Differences within the noise are ties.** On shared folds, the noise is the corrected paired
   standard error; between numbers measured separately, it is the Stage 2 noise floor. Ties go to
   the simpler, cheaper, more interpretable option. REQUIRED BACKGROUND: `ml-experiment-measurement`.
6. **Library facts are checked, not recalled.** Before relying on a version-specific API or a
   recent method, confirm it against the installed package (`pip show`, `help()`) or its docs.

## Workspace

Create a working folder at the start and keep these files current:

| File | Purpose | Created |
|---|---|---|
| `problem_statement.md` | Stage 0 framing | Stage 0 |
| `decision_log.md` | One block per stage, reasons both ways | Stage 0, appended every stage |
| `search_space.json` | Carry-forward candidates per stage, consumed by the joint search | Stage 4 onward |
| `report.md` | Skeleton written at Stage 0, filled at every gate | Stage 0 |
| `code/NN_stage.py` | One script per stage, rerunnable end to end | Every stage |

Templates for all four documents: `references/templates.md`.

**Use the bundled scripts; do not re-implement them.** Each step skill's `scripts/` folder holds
tested tools that run in seconds on typical data and encode checks ad-hoc code tends to drop: an
error cluster ten times beyond the 99th percentile, a target-derivation test robust to corrupted
rows, a significance test for informative missingness. Call them from your `code/NN_stage.py`
(by `subprocess` for the command-line ones, by import for the modules). Several stages may share
one script file when time is short. If you replace a bundled script, log which of its checks
your replacement covers.

**Write the report skeleton first.** An interrupted run must still hand over a report that says
which gates passed and what was found. Runs that leave the report to the end often run out of
time and hand over nothing.

## Stage map

| # | Stage | Load skill | Gate artifact (must exist to proceed) |
|---|---|---|---|
| 0 | Frame the problem | this skill | `problem_statement.md` complete, path chosen |
| 0b | Provenance: other files, how the data was made, achievable ceiling | this skill | provenance block in `problem_statement.md`; ceiling number or a stated reason there is none |
| 1 | Structure audit + stateless cleaning | `ds-data-audit` (part A) | column audit table, cleaning log with row counts |
| 2 | Validation design + carve test | `ds-validation-design` | split spec with rejected alternatives, frozen test IDs, noise floor from row subsamples, a second fold partition of the training portion, the carve-out cost named |
| 3 | Leakage hunt + relationship EDA (train only) | `ds-data-audit` (part B) | leak table with record-timing verdict per suspect column |
| 4 | Missing values | `ds-missing-values` | mechanism table, screened imputers, 2-3 carried forward |
| 5 | Outliers | `ds-outliers` | outlier table with error / extreme / signal verdicts, 2-3 carried forward |
| 6 | Features | `ds-feature-engineering` | feature ledger with availability time, ablations, 2-3 carried forward |
| 7 | Baselines + model families + imbalance | `ds-model-selection` | baseline table, family screen with spread, learnability verdict |
| 8 | Joint pipeline search | `ds-method-comparison` | search log, winner vs runners-up within 1 SE, locked-holdout check |
| 9 | Evaluation + sign-off | `ds-evaluation` | test score with CI, threshold, calibration, slices, error analysis, sanity tests |
| 10 | Handoff | this skill | saved pipeline scoring raw rows, model card, monitoring plan, final report |

Load each step skill when its stage starts, and follow its gate. Do not start stage N+1 until
stage N's artifact exists and its decision-log block is written.

**Forecasting path** (target is a future value of a time-indexed series): stages 0-3 as above,
with the time-based section of `ds-validation-design` (`make_split.py --scheme group_time` with a
composite `--id series,date`). Two tools change: at Stage 3 the leak hunt is the covariate
classification in `ds-forecasting` (known in advance / observed to the origin / same-period), and
`column_audit.py` still runs; `leak_screen.py` is for row-level tabular data. At Stage 8 compare
candidates with `ts_backtest.paired_compare` on shared rolling origins instead of `joint_search.py`,
applying the same one-standard-error rule. Load `ds-forecasting` for stages 4-7 and
`ds-evaluation` for stage 9.

**Unsupervised path** (clustering, anomaly detection without labels): stages 0, 1, 4, 5, 6 as
above. Replace stage 2 with a stability protocol (`ds-validation-design`, unsupervised section),
stage 7 with the unsupervised families in `ds-model-selection`, stage 9 with the unsupervised
section of `ds-evaluation`. Skip stage 3's target relationships.

## Stage 0: Frame the problem

Write `problem_statement.md` with every field filled. Where the user has not said, write the
assumption and mark it `ASSUMED`. If the user is reachable, confirm the ASSUMED lines before Stage 1.

| Field | What to write | Why it matters |
|---|---|---|
| Unit of analysis | What one row is | Sets duplicates, grouping, leakage |
| Target | Exact definition, window, source | Proxy labels cap achievable quality |
| Prediction moment | When the prediction is made | Only data available by then may be a feature |
| Decision fed | Who acts on the prediction, and how | Chooses the metric and the threshold |
| Problem type | binary / multiclass / multilabel / regression / count / ordinal / forecasting / clustering / anomaly / ranking / survival / uplift | Chooses the path and the metric family |
| Primary metric + reason | From the cost of each error type and the action capacity | Accuracy is wrong under imbalance or capacity limits |
| Secondary metrics | Calibration, interval coverage, per-slice | Catches a model good on average and bad where it matters |
| Baseline to beat | Current process, rule or heuristic | A model that loses to the rule is not shipped |
| Success threshold | The number that means "good enough" | Stops endless tuning |
| Constraints | Latency, interpretability, regulation, fairness, retraining cadence | Rules out families early |
| Data size | Rows, columns, positives, series count, history length | Chooses methods and compute |
| Time and compute budget | Wall-clock available, CPU/GPU, memory | Sizes the search |
| **Provenance** | Where the data came from, what produced it, and **every other file that shipped with it** | Decides whether a ceiling exists and whether auxiliary features exist |

Also answer: **does this need ML at all?** If a two-line rule meets the success threshold, say so.

## Stage 0b: Provenance

Three questions, answered in `problem_statement.md`. They are cheap, they are skipped almost
every time, and on derived or supplied data they are worth more than any modelling choice.

**1. What else came with the data?** List every file in the data directory, not just the ones you
were told to model. A reference table, a lookup, a dictionary, a prior year's extract, the
"original" dataset a competition was derived from - each one is a set of features waiting to be
built (membership, count, rank, distance to nearest entry, novelty) and is handed to Stage 6.
A file you did not open cannot help you.

**2. Is this data real, sampled, simulated, synthesised or augmented?** Ask, or infer it from
structure: value-frequency spikes that trace to a smaller table, impossible uniformity, absent
nulls where the source has them, truncation at both tails. If rows were generated *from* another
table, rows are not independent and the relationship to that table is itself a strong feature.
Record the answer; it changes Stage 2's split and Stage 6's menu.

**3. If the data is generated by a known or recoverable rule, what is the achievable ceiling?**
Fit the generating form (a probit / logit index, a simulation equation), estimate its noise scale,
and compute the Bayes-optimal metric analytically. Read the result carefully: a fitted index model
gives a **lower bound** on the semantic ceiling, because any effect the model cannot express is
counted as noise. Only when the generator is recovered exactly does "above the ceiling" prove
artefact exploitation; otherwise a score above it is *suspect* and the gap is the thing to
investigate. Report the ceiling and the achieved score separately, with that caveat attached.

| Situation | What to do |
|---|---|
| Rule stated in the data documentation | Fit its free parameters, compute the analytic ceiling |
| Rule unknown but data plainly synthetic | Fit the simplest plausible index model, compare its ceiling with your score |
| Real-world data | Skip, and say so: there is no ceiling to recover |

When a ceiling cannot be computed, the **grouped-split diagnostic** is the empirical substitute:
group the CV on the column that links rows back to their source, and the drop is the size of the
artefact. See `ds-validation-design`.

Skipping any of the three is a decision-log entry like any other, and "real-world data, no
generator" is a perfectly good reason.

**Compute rule.** Choose methods for the data size and problem, not for the machine. Before any
step expected to take more than a few minutes, time one fold on a sample and extrapolate. If a
method is the right choice but needs more compute than available (GPU for tabular foundation
models or deep forecasting, long searches on large data), state the requirement and ask the
user; do not silently drop it and do not silently run it for hours. If a recommended package is
not installed and installing is not permitted, log it as skipped with the install command and any
licence or hardware note, and list it in the report's next steps as a recommendation.

**Time budget.** Split the budget from `problem_statement.md` across stages: about 25% for Stages
0-3, 35% for Stages 4-7, 25% for Stage 8, 15% for Stages 9-10. **Measure time with a clock, never
estimate it**: record the start with `date` or `time.time()` at Stage 0 and write the elapsed
minutes into each gate's decision-log block. Guessed elapsed time is usually wrong in both
directions. Only when a stage has measurably overrun its share, cut breadth, not correctness: fewer
candidates, a stratified sample for screening, fewer CV repeats, a smaller search. Log each cut.
Correctness gates are never cut: the column audit and leak screen scripts (they run in seconds),
the leak table, the split, test-once, and the sanity tests.

**Compact mode** (budgets under about 45 minutes) is planned at Stage 0, not improvised later:
one script may cover several stages; 5-fold CV without repeats unless a screen is ambiguous; two
candidates per stage; 20-30 search trials with one seed. Every gate still produces its artifact,
and the decision log says compact mode was used. Compact mode shrinks **searching**, never
**looking**: these profiling scripts still run in full, since together they take under a minute:
`column_audit.py`, `leak_screen.py` (tabular), `missing_profile.py`, `outlier_profile.py`
(tabular) or `break_report` (forecasting), and `make_split.py`.

## The decision log

Append one block per stage to `decision_log.md`. This is how a reader sees that nothing was
skipped without a reason. The step skills list the candidates to consider; this block records
what happened to each.

**Menu coverage is a gate, not an aspiration.** Before closing a stage, open that step skill's
decision table and the method menu in that skill's own reference folder, and check that **every
applicable row appears in your block** as applied, carried forward, or skipped with a reason. A technique that is in the
menu and absent from the log has not been considered - it has been forgotten, and the log's claim
that nothing was skipped without a reason is then false. Write the check itself into the block:

    Menu coverage: ds-feature-engineering encoding table - one-hot (applied), ordinal (applied),
    target (carried), frequency (screened, within noise, dropped), hashing (skipped: no
    high-cardinality column). All 5 rows accounted for.

A row being *present* in the log is necessary, not sufficient: the reason must also be true for
this data. "Skipped: no high-cardinality column" is wrong if a numeric column has thousands of
distinct values that an encoder would treat as levels. Check the reason against the column audit.

This is the single cheapest guard in the whole process. Most omissions are not disagreements with
the menu; they are items nobody looked at.

```markdown
## Stage 4 - Missing values - GATE PASSED
Considered: drop rows, median, median+indicator, KNN, iterative (MICE), native NaN (GBDT)
Applied (fixed): 99999 in annual_income -> NaN (sentinel, stateless)
Carried forward: median+indicator (linear proxy log loss 0.412 +/- 0.006);
                 native NaN (GBDT proxy log loss 0.405 +/- 0.007)
Skipped: drop rows - 22% of rows affected and missingness predicts default (19% vs 11%);
         KNN - no gain over median beyond the noise floor (0.411 +/- 0.006), 30x slower;
         iterative - columns weakly correlated (max |r| 0.18), little to borrow
Evidence: code/04_missing.py, missing_profile.csv
Open risks: months_since_delinquency missing assumed to mean "never delinquent" (confirm with data owner)
```

## Carry-forward and the joint search

At stages 4-7 each step skill screens its candidates with **two proxy models**, a regularised
linear model and a small gradient-boosted tree model, under the Stage 2 CV. Two proxies because
preprocessing choices interact with model family: scaling matters to the linear proxy and not to
the tree; native missing-value handling exists only in the tree.

Each stage writes its survivors to `search_space.json`: the best two or three per applicable
family, each with its score, spread and which families it applies to. At Stage 8,
`ds-method-comparison` turns every entry into a categorical hyperparameter alongside the model's
own hyperparameters and searches the combined, conditional space. The winner is the best
combination, not the best of each stage in isolation.

Keep a candidate when it is within one standard error of the stage's best, or when it is the
best option for a family the other survivors do not serve, or - for model families at Stage 7 -
when it is cheap and sits only marginally outside the bar at library defaults, since a defaults
screen ranks defaults rather than families (see `ds-model-selection`). Drop it with a logged
reason otherwise. The same rule is stated in `ds-method-comparison` and `ds-model-selection`;
if they disagree, this paragraph governs.

## Stop signals

Stop and report rather than pressing on when any of these appear:

- **Too good to be true.** A score far above domain expectations, the business baseline, or
  published results for similar problems. Return to Stage 3; the usual cause is a column recorded
  after the outcome.
- **Not learnable yet.** At Stage 7 the best family does not beat the simple baseline by more
  than the noise floor, and the learning curve is flat. More algorithms will not help. Report
  it and recommend better features, labels or data instead.
- **Split cannot be made honest.** Groups or time structure that no split scheme respects with
  the data available. Report it before modelling.
- **Budget exhausted.** Finish the current gate, fill the report with what passed, list the
  queue that remains.

## Red flags - stop and return to the named stage

| Thought | Reality | Go to |
|---|---|---|
| "AUC 0.97, great" | Check the leak table before celebrating | 3 |
| "I'll impute on the full data, it's just median" | The median is learned from test rows | 1, 4 |
| "Random K-fold is fine, rows look independent" | Check for entity IDs and dates first | 2 |
| "This column is the strongest predictor" | Ask when it is recorded relative to the outcome | 3 |
| "Skip the baseline, boosting always wins" | On small or mostly linear signals, regularised linear models often win | 7 |
| "Run all the heavy models now, screen later" | Screen cheap first, then spend compute on survivors | 7 |
| "Model B scores 0.004 higher" | Compare against the noise floor | 8 |
| "Quick look at the test set to pick a threshold" | Thresholds are chosen on validation | 9 |
| "SMOTE the data, then split" | Resampling happens inside the training fold | 7 |
| "Not worth writing why I skipped it" | Unreasoned skips are how steps get missed | log |
| "I'll write the report at the end" | Interrupted runs then deliver nothing | 0 |
| "I only modelled the file I was told to" | List every file that shipped with the data; auxiliary tables are features | 0b |
| "It's synthetic, so the ceiling is 1.0" | Recover the generator and compute the real ceiling before celebrating | 0b |
| "The menu is long, I screened the obvious ones" | An item absent from the log was forgotten, not considered | log |
| "It saved without error" | Saving is not loading. Reload it in a fresh process | 10 |

## Stage 10: Handoff

- Save the whole fitted pipeline, preprocessing included, with pinned library versions and seeds.
  Its first step must be the Stage 1 stateless cleaning rules (sentinels, spellings, impossible
  values), so raw production rows get exactly the cleaning the training data got.
- Score a handful of raw, uncleaned rows end to end through the saved pipeline; confirm output
  and latency. Feed one row twice and confirm identical scores.
- **Reload the saved file in a fresh process and reproduce a prediction.** Saving is not the test;
  loading is. A pipeline holding a closure, a lambda, or a function defined in `__main__` will
  pickle without complaint and then fail to load anywhere else. Every transformer that ends up in
  the artifact must be a module-level class or function, with its configuration as instance state.
  Run this check the first time a custom transformer is written, not at Stage 10 when the design
  is already locked.
- Write the model card and monitoring plan (`ds-evaluation` has both).
- Fill `report.md` from the template: problem, data found, decisions with reasons, results
  against baselines with intervals, error analysis, limitations, next steps.

## Common mistakes

| Mistake | Fix |
|---|---|
| Gates answered with prose | Every gate names a file or table; link it in the decision log |
| Carry-forward of one option | Keep 2-3 unless one dominates beyond the noise floor on both proxies |
| Searching over leaky columns | Correctness choices are fixed before the search |
| Spending the budget on tuning | Most gain comes from framing, leakage and features; tune last |
| Treating ASSUMED lines as facts | List them in the report's limitations and ask the data owner |
