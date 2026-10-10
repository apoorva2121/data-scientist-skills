# Missing values: full method menu

Package versions and release dates from PyPI's JSON API on 2026-09-27. scikit-learn behaviour
confirmed on the installed 1.7.2. "Maintained" means a release within roughly the last three years.

## Diagnosis

| Method | How | Notes |
|---|---|---|
| Missing share and pattern | `df.isna().mean()`, co-missing Jaccard | `scripts/missing_profile.py` |
| Predict the indicator from observed columns | CV AUC of a classifier for `is_missing` | Near 0.5 is consistent with MCAR; high means MAR-like (or structural) |
| Indicator vs target | AUC or Spearman of the indicator alone | Significant means informative; see `ds-data-audit` leak screen for the Hanley-McNeil test |
| Missingness over time | Share per period | Detects collection changes and train-serving differences |
| Little's MCAR test | Classical chi-square test of MCAR | Sensitive to sample size and distributional assumptions; the predict-the-indicator check is more practical for prediction work |
| Visual | missingno matrix, heatmap, dendrogram | `missingno` 0.5.2 (2023-02-26), unmaintained but stable |

## Deletion

| Method | Use when | Risk |
|---|---|---|
| Listwise (drop rows) | Few rows, MCAR-like, never missing at inference | Bias if not MCAR; does not solve inference-time missing values |
| Drop column | Very high missing share and not informative | Loses signal if informative |
| Pairwise | Correlation or covariance estimation only | Inconsistent matrices; not for model training |

## Single imputation (scikit-learn 1.7.2)

| Method | Class | Mechanism suited | Families | Caveats |
|---|---|---|---|---|
| Mean / median / most frequent / constant | `SimpleImputer` | MCAR-like | all | Shrinks variance; median is robust to skew |
| Missing indicator | `MissingIndicator`, or `add_indicator=True` on imputers | informative, any | all | Default adds indicators only for columns missing in training |
| k-nearest neighbours | `KNNImputer` | MAR-like | linear, distance, neural | Scale first; slow on large data (screen on a sample) |
| Iterative, Bayesian ridge (MICE-style chained equations) | `IterativeImputer` (needs `sklearn.experimental.enable_iterative_imputer`) | MAR-like | linear, neural | Assumes roughly linear relations between columns |
| Iterative with trees (MissForest-style) | `IterativeImputer(estimator=ExtraTreesRegressor(...))` | MAR-like, non-linear | all | Slow; can overfit small data |
| Native NaN handling | LightGBM, XGBoost, CatBoost, `HistGradientBoosting*` | any, including informative | boosted trees | Learns a default split direction for missing values |

## Specialised packages

| Package | pip name | Latest (released) | Maintained? | Use when |
|---|---|---|---|---|
| miceforest | `miceforest` | 6.0.5 (2025-10-27) | yes | Fast LightGBM-based MICE with multiple datasets; multiple imputation |
| MissForest | `missforest` | 4.2.3 (2025-01-03) | yes | Random-forest iterative imputation as a packaged class |
| statsmodels MICE | `statsmodels` | 0.15.0 (2026-08-27) | yes | Multiple imputation for regression inference with pooled estimates |
| Autoimpute | `autoimpute` | 0.14.1 (2023-12-30) | borderline | Multiple imputation with analysis pooling |
| HyperImpute | `hyperimpute` | 0.1.17 (2023-02-28) | no | Automatic imputer selection incl. GAIN, MIWAE; research use only |
| fancyimpute | `fancyimpute` | 0.7.0 (2021-10-21) | no | SoftImpute, matrix factorisation; prefer maintained alternatives |
| missingpy | `missingpy` | 0.2.0 (2018-12-10) | no | Do not use: no release since 2018; use `IterativeImputer` with trees or `missforest` instead |

## Generative and deep imputers (advanced)

GAIN (adversarial), MIWAE and not-MIWAE (variational, the latter models MNAR), denoising
autoencoders, diffusion-based imputers. They can model complex joint distributions on large data,
but for prediction on tabular data they rarely beat boosted trees with native NaN plus indicators,
and the maintained packaging is thin (see HyperImpute above). Use them only when screening shows a
gain above the noise floor and the compute is available; say so in the decision log.

## Multiple imputation

Create m completed datasets (m = 5-20), fit the analysis on each, pool with Rubin's rules: the
pooled estimate is the mean of the m estimates; the total variance is the within-imputation variance
plus (1 + 1/m) times the between-imputation variance. Needed for honest standard errors of
coefficients. For prediction, averaging models across imputations is a form of ensembling; accept
it only if the gain passes the noise floor.

## Time series

| Situation | Method | Leak note |
|---|---|---|
| Feature gaps | Forward fill (last observation carried forward), seasonal fill from last season | Uses only past values |
| Target history gaps before the forecast origin | Linear or seasonal interpolation inside the training window | Never interpolate across the origin |
| Regular grid needed | Reindex to the full calendar first, then treat created gaps explicitly | Missing periods are otherwise invisible |
| Intermittent demand (true zeros) | Not missing: keep zeros | Confusing zeros with missing distorts forecasts |
