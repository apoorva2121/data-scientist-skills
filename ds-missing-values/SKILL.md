---
name: ds-missing-values
description: Use when a tabular or time-series dataset has missing values, NaN, nulls or blanks and a model must be built on it; when choosing between dropping rows, mean/median/mode imputation, KNN, iterative (MICE) imputation, missing indicators or a model's native missing-value handling; when deciding whether missingness is random or informative; or when imputation may have leaked information across a train/test split.
---

# Missing Values

## Overview

Diagnose why values are missing, then choose. The same 20% missing can call for a constant, an
indicator, a model-based imputer or nothing at all, depending on the mechanism, the model family
and whether missingness itself predicts the target.

Three rules frame everything here:
1. **Imputers are fitted in-fold**, inside the pipeline. A median computed on all rows has seen
   the test set. (Converting sentinel codes to null was a stateless Stage 1 step.)
2. **The model will meet missing values at inference.** Dropping training rows does not remove
   missing values from tomorrow's data, so the pipeline must handle them either way.
3. **Informative missingness is signal.** If "value is missing" predicts the target, imputing it
   away without an indicator throws that signal out.

## Step 1: Diagnose (training rows only)

```bash
python scripts/missing_profile.py train.csv --target <t> --task binary|multiclass|regression \
    --exclude <ids> [--time <date col>] --out missing_profile.csv
```

Per column it reports the missing share, whether missingness is **predictable from the other
columns** (CV AUC), whether the **indicator predicts the target**, which columns go missing
**together**, and the **trend over time**. Read it with this table:

| Evidence | Likely mechanism | What it means |
|---|---|---|
| Not predictable from other columns (AUC under about 0.6), no target signal | MCAR-like | Any reasonable imputer works; simple is fine |
| Predictable from other columns, no target signal | MAR-like | Impute from other columns (KNN, iterative); indicator optional |
| Indicator predicts the target | informative | Keep an indicator, or native NaN handling in boosted trees |
| Missing together with related columns, a clear "does not apply" reason | structural | Not missing data: fill with a meaningful constant plus indicator |
| Share changes sharply over time | collection change | Check it will look the same at serving time; consider a period feature |
| Depends on the unseen value itself (high incomes left blank) | MNAR | Cannot be proven from data; domain knowledge. Indicator plus sensitivity check |

A column can be several at once: missingness can be predictable from other columns and also
informative about the target. When it is predictable from the same columns that drive the target,
the indicator may add nothing beyond them; the ablation in Step 3 settles it.

## Step 2: Choose candidates

Pick from this table, then screen them (Step 3). The full menu, including advanced and
generative imputers and their maintenance status, is in `references/methods.md`.

| Situation | Candidates for linear / distance / neural models | Candidates for boosted trees |
|---|---|---|
| MCAR-like, under about 5% | median (numeric), most-frequent or a "missing" level (categorical) | native NaN |
| MCAR-like, 5-40% | median + indicator; KNN | native NaN; median + indicator |
| MAR-like | KNN (after scaling); iterative (Bayesian ridge, or extra-trees for non-linear) + indicator | native NaN; iterative + indicator |
| Informative | median + indicator (the indicator is the point) | native NaN (learns the missing direction); median + indicator |
| Structural | constant (0, "none") + indicator | same |
| Over about 60-80% missing | drop the column, unless informative: then indicator only | native NaN or indicator only |
| Categorical columns | "missing" as its own level; rarely most-frequent | native categorical handling, or "missing" level |
| Time series, feature history | forward fill from past values only | same, or native NaN |
| Time series, the target itself | see `ds-forecasting` (gaps, reindexing, intermittent demand) | |

**Dropping rows** is a candidate only when all hold: few rows, MCAR-like, the missing column is
never missing at inference or the pipeline still imputes it there, and dropped rows are not
concentrated in one class or period. Say which held.

**Why not always the fanciest imputer:** prediction quality rarely improves beyond a good simple
imputer plus an indicator, especially for boosted trees with native handling. Model-based
imputers cost time and can overfit on small data. Screen them; do not assume them.

**Iterative imputation in scikit-learn 1.7.2** is still experimental and needs its enable import:

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
mice_linear = IterativeImputer(max_iter=10, random_state=0, add_indicator=True)
mice_trees = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, min_samples_leaf=5, random_state=0),
                              max_iter=5, random_state=0, add_indicator=True)
```

## Step 3: Screen and carry forward

Build one pipeline per candidate and family, identical apart from the imputer, and screen with
`ds-method-comparison` (two proxies, the Stage 2 folds, the noise floor). Carry forward the
survivors per family to `search_space.json`, simplest first, for example:

```json
"missing_values": [
  {"name": "median_indicator", "applies_to": ["linear", "gbdt"], "proxy_scores": {"linear": [-0.412, 0.006], "gbdt": [-0.407, 0.007]}},
  {"name": "knn", "applies_to": ["linear"], "proxy_scores": {"linear": [-0.411, 0.006]}},
  {"name": "native_nan", "applies_to": ["gbdt"], "proxy_scores": {"gbdt": [-0.405, 0.007]}}
]
```

## Step 4: Checks on the chosen candidates

- **In-fold:** the imputer is a pipeline step, fitted by `fit` on the training fold. No `fillna`
  with a statistic computed outside the pipeline.
- **Distortion:** for each imputed numeric column, compare the variance and its correlation with
  the top features before and after imputation on a training fold. Mean and median imputation
  shrink variance and pull correlations toward zero; a large change argues for KNN or iterative.
- **Indicator value:** screen with and without the indicator. Keep it when the difference passes
  the noise floor or the diagnosis says informative.
- **Inference readiness:** push one row with every feature missing, and one with a missing value in
  a column that had none in training, through the fitted pipeline. Both must return a prediction.
- **Multiple imputation** is for statistical inference (Rubin's rules for pooled coefficients and
  standard errors), not usually for prediction. If the deliverable includes effect sizes or
  confidence intervals on coefficients, use it (`statsmodels`, `miceforest`) and pool properly.

## Gate (all must exist)

- `missing_profile.csv` with a mechanism verdict and an action for every column with missing values
- screening tables per family, and the `missing_values` entry in `search_space.json`
- inference-readiness check passed on the chosen candidates
- decision-log block: considered, carried forward, skipped with reasons, open MNAR assumptions

## Red flags

| Thought | Reality |
|---|---|
| "Just df.fillna(df.median())" | That median was computed on test rows too |
| "Drop rows with NaN, it's only 8%" | Tomorrow's data still has NaN; and was that 8% random? |
| "MICE is more sophisticated, so better" | Screen it; boosted trees with native NaN often match it at a fraction of the cost |
| "No need for an indicator, I imputed it" | If missingness predicts the target, imputation erased the signal |
| "Most-frequent category for missing categoricals" | A separate "missing" level keeps the information |
| "Interpolate the gaps in this feature" | Interpolation uses future values; for features, fill from the past only |
