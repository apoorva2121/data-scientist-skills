---
name: ds-evaluation
description: Use when reporting how good a tabular or time-series model really is - scoring the held-out test set, confidence intervals on a metric, choosing a decision threshold, probability calibration, prediction intervals or conformal prediction, performance per segment or fairness across groups, error analysis, feature importance, SHAP or partial dependence explanations, sanity checks, model cards, drift monitoring, or saving a model for production.
---

# Evaluation and Sign-off

## Overview

Stage 9 produces the number the business will rely on, and the evidence behind it. Its central
rule: **the test set is scored once**, with a pipeline and threshold that were fixed before
looking. Everything that involves a choice (threshold, calibration method, which model) is
decided on validation predictions first.

If the test score disappoints, report it. Changing anything after seeing the test score turns the
test set into a validation set; from then on, only a fresh untouched sample can give an honest
estimate, and the report must say so.

## Step 1: Preconditions

- The Stage 8 pick is refitted on train plus holdout with fixed hyperparameters.
- The threshold and any calibrator are chosen on out-of-fold or holdout predictions (Steps 2-3).
- `verify_test_untouched("split")` from `ds-validation-design` returns True.
- The metrics to report were fixed at Stage 0; add secondary metrics, never swap the primary.

## Step 2: Threshold (classification, decided on validation)

| Objective | Threshold |
|---|---|
| Action capacity ("call the top 10%") | The score at that rank: `capacity_threshold(scores, 0.10)`; report precision and recall at it |
| Known costs of false positives and false negatives | Minimise expected cost on validation predictions |
| F-beta, balanced accuracy, or another metric | `sklearn.model_selection.TunedThresholdClassifierCV` |
| Scores consumed as probabilities downstream | No threshold; calibrate instead (Step 3) |

## Step 3: Calibration (when probabilities are used as probabilities)

Check a reliability table (predicted vs observed per score decile), expected calibration error
and the Brier score with `calibration_table`. Class weights, resampling and many models distort
probabilities. If calibration is off, fit a calibrator on held-out predictions, never on test:

| Method | Use when |
|---|---|
| Sigmoid (Platt) | Small calibration sets, S-shaped distortion |
| Isotonic | About 1,000 or more calibration rows; any monotonic distortion |
| Venn-ABERS (`venn-abers`) | Calibrated probability intervals with validity guarantees |
| Temperature or beta scaling (`netcal`) | Neural nets; asymmetric distortion |

In scikit-learn 1.7.2, calibrate an already-fitted model with
`CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")` fitted on the holdout
(`from sklearn.frozen import FrozenEstimator`), or let `CalibratedClassifierCV(pipe, cv=folds)` cross-fit.

## Step 4: Uncertainty for each prediction

| Need | Method |
|---|---|
| Interval around a regression prediction | Conformal: `mapie.regression.SplitConformalRegressor`, `CrossConformalRegressor`; or `ConformalizedQuantileRegressor` for adaptive widths |
| Intervals for time series | `mapie.regression.TimeSeriesRegressor`, or conformal on backtest residuals per horizon (`ds-forecasting`) |
| Prediction sets in classification | `mapie.classification.SplitConformalClassifier` |
| Native quantiles | Quantile objectives in LightGBM, XGBoost, CatBoost; `ngboost` distributions |

MAPIE 1.x replaced the old `MapieRegressor` API; the class names above are from the 1.5.0 source.
Report **empirical coverage** and mean width on the test set with `interval_coverage`.

## Step 5: Score the test set, once

```python
import sys; sys.path.insert(0, "<path to ds-evaluation/scripts>")
from final_eval import bootstrap_ci, confusion_at, calibration_table, slice_table, interval_coverage
point, lo, hi = bootstrap_ci(y_test, scores, roc_auc_score)
```

Report every metric with its 95% bootstrap interval, next to the baselines scored on the same test
rows. CV mean, holdout and test should agree within their intervals; that agreement is evidence no
leak survived. A test score far above the business baseline, published results for similar
problems, or the CV estimate is a stop signal, not a success.

## Step 6: Slices and fairness

Compute the primary metric per segment named in Stage 3 notes and the problem statement (region,
product, customer tenure band, period) with `slice_table`, which adds intervals and skips tiny
groups. Name the worst slice in the report. Where protected attributes exist, report group metrics
with `fairlearn` (`MetricFrame`, `demographic_parity_difference`, `equalized_odds_difference`); in
regulated settings this is a requirement, not an extra. Mitigation (`ThresholdOptimizer`, reweighting)
is a business and legal decision: present the trade-off, do not choose silently.

## Step 7: Error analysis

- Read the 20 worst errors by hand: highest-scored negatives, lowest-scored positives, largest residuals.
- Plot residuals against the prediction and against each top feature; look for curvature and funnels.
- Multiclass: the most-confused class pairs.
- Write each pattern as a next step (a missing feature, a data problem, a segment needing its own model).
  These are for the next iteration on new data, not for re-tuning against this test set.

## Step 8: Sanity tests

| Test | Pass condition |
|---|---|
| Shuffled labels (`shuffled_label_check`, on the train folds) | Score falls to the trivial baseline (for PR-AUC, about the positive rate) |
| Duplicate row | Identical prediction twice |
| Raw rows end to end | Uncleaned rows (sentinels, odd spellings, missing values) score without error through the saved pipeline |
| **Reload in a fresh process** | `joblib.load` in a new interpreter, then reproduce a saved prediction exactly. Saving is not the test - a closure, a lambda, or a function defined in `__main__` pickles without complaint and fails to load anywhere else |
| Directional expectations | Changing a feature in a known direction moves the prediction the plausible way |
| Too good to be true | No metric far above domain expectations, the business baseline or the CV estimate |

## Step 9: Explanations

- Global: permutation importance on held-out data (`sklearn.inspection.permutation_importance`);
  SHAP (`shap.TreeExplainer` for trees, `PermutationExplainer` otherwise). With correlated features,
  permute or explain **groups** of features together, since correlated features share credit.
- Shape of effects: partial dependence and ICE (`PartialDependenceDisplay`); ALE when features are
  correlated. Interactions: `shapiq`.
- Local: SHAP values for a handful of real cases, including errors.
- Check the top features against the Stage 3 leak table. A leak-suspect column at the top sends
  the project back to Stage 3.

## Step 10: Hand-off artefacts

- **Persistence:** `skops.io` for scikit-learn pipelines (pickle and joblib can execute arbitrary
  code on load; only load files you trust); ONNX via `skl2onnx` for cross-platform serving. Pin
  library versions next to the file.
- **Model card:** `references/model-card.md` template: intended use, data, metrics with intervals,
  slices, limitations, ethical considerations, monitoring.
- **Monitoring plan:** per-feature drift (PSI, KS, Jensen-Shannon; `evidently`), prediction
  distribution drift, delayed-label performance tracking, label-free performance estimation for
  classifiers (`nannyml` CBPE), data validation of each batch with the Stage 1 rules (`pandera`),
  and a stated retraining trigger.

## Gate (all must exist)

- test-untouched check passed; threshold and calibrator chosen before test scoring
- test metrics with bootstrap intervals, next to baselines on the same rows
- calibration table (if probabilities are consumed) and interval coverage (if intervals are shipped)
- slice table with the worst slice named; fairness table where protected attributes exist
- error-analysis notes, sanity-test results, global and local explanations
- saved pipeline, model card and monitoring plan; final decision-log block

## Red flags

| Thought | Reality |
|---|---|
| "Pick the threshold that maximises F1 on test" | Then the reported F1 is optimistic; choose on validation |
| "Test AUC is 0.97" | Compare with the CV estimate and the business baseline before celebrating |
| "Recalibrate on the test set" | The calibrator has then seen test labels |
| "Test score is low, let me tweak and rescore" | The test set becomes a validation set; say so or get new data |
| "Overall accuracy is fine" | Report the worst slice; averages hide failing groups |
| "SHAP says this feature does not matter" | With correlated features, credit is split; explain groups |
| "pickle.dump and ship it" | Unpinned, insecure to load from untrusted sources; use skops and pin versions |
