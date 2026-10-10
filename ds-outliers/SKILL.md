---
name: ds-outliers
description: Use when a tabular or time-series dataset has extreme values, heavy tails, suspicious spikes or data-entry errors; when deciding whether to remove, cap (winsorize), transform or keep outliers before modelling; when a regression target has extreme values; when choosing between IQR, z-score, MAD, Isolation Forest, LOF or other detectors; or when the task itself is anomaly, fraud or fault detection.
---

# Outliers

## Overview

"Outlier" covers three different things, and each needs a different action:

| Kind | Example | Action |
|---|---|---|
| **Error** | A birth year of 1850, a weight entered in grams instead of kilograms | Fix or null it (then Stage 4 imputes it) |
| **Genuine extreme** | One very large wholesale order in a retail table | Keep; limit its leverage for sensitive models |
| **Signal** | Unusually large transactions in a fraud problem | Keep; possibly make it a feature |

Removing a genuine extreme biases the model toward the middle. Removing a signal outlier deletes
the thing you are trying to predict. Only errors are removed, and only on evidence.

Rules: thresholds that are learned from data (quantiles, MAD, detector scores) are fitted in-fold.
Rows are never removed from the test set, since tomorrow's data will contain the same extremes.
Tree-based models barely notice feature outliers; linear, distance-based and neural models do.

## Step 1: Profile (training rows only)

```bash
python scripts/outlier_profile.py train.csv --target <t> --task binary|multiclass|regression \
    --exclude <ids> [--time <date col>] --out outlier_profile.csv
```

Per numeric column it counts rows beyond 1.5 and 3 IQR and beyond 3.5 robust SDs (MAD), then
compares the **target rate among the extremes with the overall rate** (the lift). It also checks
whether the extremes cluster in one period and whether a group sits beyond 10x the 99th percentile.
It finishes with a multivariate Isolation Forest pass, since a row can be unusual only in
combination, such as a small flat with a huge price.

## Step 2: Verdict per column

| Evidence | Verdict | Action |
|---|---|---|
| Violates a domain rule (impossible age, negative quantity, date before founding) | error | Stateless null at Stage 1; if missed there, go back and do it there |
| A cluster about 10x, 100x or 1000x beyond the 99th percentile | likely unit or entry error | Verify a sample by hand; then `FarValueNuller` in-fold, or a domain rule |
| Lift far from 1 with enough rows (roughly 10 or more) | signal | Keep; candidate `AnomalyScore` or "is_extreme" feature at Stage 6 |
| Heavy right tail (skew above about 1), many rows beyond 1.5 IQR | genuine tail | Transform at Stage 6 (log, Yeo-Johnson), or winsorize for linear models |
| A few isolated rows, no rule broken | undecided | Inspect each; default to keeping them |
| Extremes clustered in one period | collection issue | Check with the data owner; may be a pipeline change, not reality |
| The target itself has extremes (regression) | target tail | See Step 3; never drop test rows |

Write the verdict and the evidence per column in `outlier_profile.csv` (add a `verdict` column).

## Step 3: Treatment candidates by family

| Family | Candidates to screen | Why |
|---|---|---|
| Boosted trees, random forests | `none` (reference) | Splits depend on order, not distance; capping rarely changes them |
| Linear, GLM | `none`, `Winsorizer(0.01, 0.99)`, `RobustClipper(3.5)`, log or Yeo-Johnson on the tailed columns | One extreme row can tilt a least-squares fit |
| Distance-based (kNN, SVM, clustering) | as linear, plus `RobustScaler` instead of `StandardScaler` | Distances are dominated by extreme coordinates |
| Neural networks | Winsorize or quantile-transform inputs | Extreme inputs saturate activations and slow training |
| Any, extremes are signal | `AnomalyScore` appended as a feature | Lets the model use "how unusual" directly |

Treatments live in `scripts/outlier_transformers.py` (all fitted in-fold, NaN passes through):
`Winsorizer`, `RobustClipper`, `FarValueNuller`, `AnomalyScore`.

**Target outliers in regression.** Keep them in evaluation. For training, choose between:
a log or Box-Cox target (Stage 6, back-transform with bias correction), a robust loss (Huber,
quantile or absolute error, available in LightGBM, XGBoost, CatBoost and `HuberRegressor`), or
removal of confirmed target errors only. The metric decides too: RMSE punishes the tail, MAE does not.

**Row removal** is an option only for confirmed errors that cannot be nulled cell by cell. Report
the count and check that removed rows are not concentrated in one class, segment or period.

## Step 4: Screen and carry forward

Screen the candidates per family with `ds-method-comparison` (same folds, noise floor). Often no
treatment separates from "none" beyond noise: that is a normal outcome, and the tie goes to `none`.
When a suspected error cluster could not be verified (no data dictionary, no owner), do not simply
defer it: screen `none` against `FarValueNuller` and carry both forward if they tie, so the joint
search and the Stage 9 slice check decide on evidence. Carry forward the survivors, simplest first:

```json
"outliers": [
  {"name": "none", "applies_to": ["linear", "gbdt"]},
  {"name": "winsorize_p01_p99", "applies_to": ["linear"]}
]
```

## When the task is anomaly detection

If the goal itself is finding anomalies (fraud, faults, intrusions) rather than cleaning data:

- **With some labels:** treat it as highly imbalanced classification (`ds-model-selection`);
  semi-supervised detectors that use a few labels usually beat fully unsupervised ones.
- **Without labels:** choose detectors by data shape (the menu is in `references/methods.md`),
  combine several (average of rank-normalised scores), and check stability across subsamples
  (`ds-validation-design`, unsupervised section). Validate the top-ranked rows with a domain expert.
- **PyOD 3** (3.6.6, released 2026-09-17) ships 61 detectors across tabular, time-series and other
  data, an end-to-end `ADEngine`, and its own Claude Code skill (`od-expert`, installed with
  `pyod install skill`). For a dedicated anomaly-detection project, suggest it to the user.

## Gate (all must exist)

- `outlier_profile.csv` with a verdict and evidence for every flagged column, target included
- counts of rows or cells nulled or removed, and their class and period concentration
- screening tables and the `outliers` entry in `search_space.json`
- decision-log block, including every treatment skipped and why

## Red flags

| Thought | Reality |
|---|---|
| "Remove everything beyond 3 SD" | That removes genuine extremes and signal along with errors |
| "Drop outliers from the whole dataset" | The test set must keep them; production data will have them |
| "Cap using quantiles of the full data" | Quantiles are learned in-fold, like any other statistic |
| "Trees need winsorizing too" | Screen it; usually no effect, so the tie goes to none |
| "These big transactions look wrong" | In fraud detection they are the point; check the lift first |
| "Log the target and report RMSE on logs" | Back-transform before reporting, with bias correction |
