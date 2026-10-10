# Data Scientist Skill: Workflow Inventory (working draft)

Purpose: list every stage from "here is data + a motive" to "here is a validated model",
what to do at each stage, what to check and how, and how the method is chosen.
This is the raw material for deciding skill boundaries. Nothing here is a skill yet.

Status: draft v1, 2026-09-27. Open question: tabular + time series only, or also NLP / CV?

---

## Stage 0. Problem framing (before touching data)

Do
- Define the unit of analysis (what is one row), the target, the decision the prediction feeds,
  the action taken on it, and the moment of prediction. Only information available at that
  moment may become a feature.
- Classify the problem type: binary / multiclass / multilabel classification, regression,
  count or ordinal, time-series forecasting, clustering, anomaly detection,
  ranking / recommendation, survival (time-to-event), uplift / causal.
- Pick the offline metric from the cost of each error type. Name the baseline to beat
  (current process, a rule, a heuristic).
- Record constraints: latency, interpretability, regulatory / fairness, retraining cadence,
  data available at inference.

Check
- Is ML needed at all, or does a rule solve it?
- Is the label real or a proxy? How noisy is it?
- Does the offline metric actually track the business metric?

Output: one-page problem statement (target definition, metric, baseline, constraints).

## Stage 1. Data audit (structure-level EDA)

Do
- Provenance per column: source, when recorded, join keys, grain.
- Schema: dtypes, row count vs expected, unique keys, duplicates, cardinality per column.
- Target: distribution, class balance, range, impossible values, label noise.
- Missingness map: share missing per column, patterns across columns, and whether a
  missing-indicator correlates with the target.
- Time coverage: date range, gaps, target rate over time.

Check (leakage hunt)
- Columns recorded after the outcome or after the decision moment.
- IDs or codes that encode the target.
- Any single feature with suspiciously high univariate predictive power.
- Rows duplicated across what will become train and test.
- Train / serving skew: does this data look like what will be scored later?

## Stage 2. Validation design (before cleaning, so nothing fitted on test leaks back)

Do
- Carve the hold-out test set first. It is touched once, at the very end.
- Choose the split by data structure:
  - stratified random k-fold: independent rows
  - GroupKFold: several rows per entity (customer, patient, device)
  - time-based / rolling-origin: forecasting, or any data that drifts over time
  - nested CV: small data where tuning would overfit the validation folds
  - repeated CV: small data, to estimate the noise floor

Check
- No entity or time period on both sides of a split.
- Target distribution similar across folds.
- Test set large enough to detect the metric differences that matter.

Rule for every later stage: fit transformers on the training fold only, inside a pipeline.

## Stage 3. Cleaning

Do
- Fix types, parse dates, normalise category spellings / case / whitespace, unify units.
- Replace sentinel values (-999, 9999, "N/A", "", "unknown") with real nulls.
- Remove exact and near duplicates after deciding which copy is authoritative.
- Fix or flag impossible values and impossible combinations.

Check
- Log row count after each step.
- Dropped rows are not concentrated in one class, segment, or period.

## Stage 4. Missing value treatment

Diagnose first
- Mechanism: MCAR / MAR / MNAR. Compare other columns for rows with vs without the value.
  Test whether a missing indicator predicts the target.

Methods
- Drop rows: only when few and missing at random.
- Drop column: mostly empty and uninformative.
- Median / mode: simple, robust, default for tree models.
- Constant + indicator column: when missingness is informative.
- KNN / iterative (MICE-style) imputation: linear models, moderate missingness, correlated features.
- Forward fill / interpolation: time series.
- Native NaN handling: LightGBM, XGBoost, CatBoost, HistGradientBoosting.

Choose by: mechanism, share missing, model family, whether missingness is informative.

Check
- Imputer fitted on train fold only.
- Distributions before vs after imputation.
- CV metric with vs without the indicator column.

## Stage 5. Outlier detection and treatment

Detect
- Univariate: IQR fences, MAD-based robust z-score, percentile caps.
- Multivariate: Isolation Forest, LOF, Mahalanobis distance, DBSCAN noise points.
- Model-based: large residuals from a simple model.
- Domain rules beat statistics wherever they exist.

Decide
- Error (fix or remove) vs genuine extreme (keep, maybe transform) vs the thing you are
  predicting (fraud, failures: never remove).
- Outlier in the target vs in a feature is a different decision.

Treat
- Remove, winsorise / cap, log transform, robust scaler, robust loss (Huber), or use tree
  models that are insensitive to feature outliers.

Check
- How many rows affected, and whether they cluster in one class or period.
- CV metric with vs without treatment.
- Never remove rows from the test set.

## Stage 6. Relationship EDA

Do
- Univariate shape per feature (skew, multimodality).
- Bivariate vs target: binned target rate for numerics, target rate per level with counts
  for categoricals.
- Correlation matrix and VIF for multicollinearity.
- Interactions and target trend over time.

Check
- Near-zero variance features.
- Correlated pairs above ~0.9.
- High-cardinality categoricals, rare levels.
- Target relationship stable across time and segments.

Output: feature ideas and drop candidates.

## Stage 7. Feature engineering and selection

Encode
- Categorical: one-hot (low cardinality), ordinal (ordered), out-of-fold target / WOE
  encoding (high cardinality), frequency, hashing, native categorical support
  (CatBoost, LightGBM).
- Numeric: log / Box-Cox / Yeo-Johnson for skew (linear models), ratios, differences,
  binning, polynomial terms.
- Scaling: Standard / MinMax / Robust. Needed for linear, SVM, KNN, NN. Not for trees.
- Date / time: cyclical encoding, lags, rolling stats, time since last event, holidays.
- Relational: group-by aggregates computed only from the training window.
- Text: TF-IDF, embeddings. Domain features from the problem statement.

Select
- Filter: variance, correlation, mutual information, chi-square.
- Embedded: L1, tree importance.
- Wrapper: recursive feature elimination, Boruta.
- Permutation importance on validation folds.

Check
- Every encoder fitted in-fold (target encoding especially).
- Feature count reasonable for row count.
- Importance stable across folds.
- Ablation per feature group: CV metric with vs without.

## Stage 8. Imbalance and target transforms (when applicable)

Classification, in order of preference
- Do nothing and tune the decision threshold.
- Class weights.
- Resampling (undersample, SMOTE) on the training fold only.
- Focal loss.
- Reframe as anomaly detection when extremely rare.

Regression
- Log / Box-Cox the target when skewed; back-transform with bias correction.

Check
- Resampling never touches validation or test.
- Recalibrate probabilities after reweighting.
- Metric is PR-AUC / F-beta, never accuracy, when imbalanced.

## Stage 9. Baselines

- Trivial: majority class, mean, last value, seasonal naive.
- Simple: regularised logistic / linear regression, one shallow decision tree.
- Business: the existing rule or process.

Everything later must beat these by more than the noise floor.

## Stage 10. Candidate models by problem type

- Tabular classification / regression: regularised linear, random forest, gradient boosting
  (LightGBM, XGBoost, CatBoost; usually strongest), SVM / KNN on small data, TabPFN on tiny
  data, neural nets only with large data.
- Forecasting: seasonal naive, ETS, ARIMA / SARIMAX, Prophet, gradient boosting on lag
  features, global deep models (N-BEATS, TFT) when there are many series.
- Clustering: k-means, GMM, hierarchical, DBSCAN / HDBSCAN. Choose by cluster shape,
  density, scale.
- Anomaly: Isolation Forest, one-class SVM, LOF, autoencoders.
- Ranking / recommendation: LightGBM ranker, matrix factorisation, two-tower.
- Survival: Cox PH, random survival forest, gradient boosting survival.
- Uplift: T / S / X learners, uplift trees.

Selection criteria: data size, feature types, interpretability, latency, training cost,
need for calibrated probabilities, monotonic constraints.

Process: run 3 to 5 families with defaults under the same CV, compare mean and spread,
carry the top 1 or 2 forward.

## Stage 11. Hyperparameter tuning

- Random search or Bayesian (Optuna); successive halving / Hyperband for expensive models;
  early stopping for boosting. Grid only for tiny spaces.
- Tune on validation folds, never on test. Many trials overfit the validation folds:
  use nested CV or a separate holdout on small data.

Check
- Gain over defaults exceeds fold-to-fold spread.
- Learning curves (train vs validation, and metric vs training size).

## Stage 12. Evaluation and error analysis

Metrics by task
- Classification: PR-AUC, ROC-AUC, log loss, Brier, F-beta, MCC, confusion matrix at the
  chosen threshold. Accuracy only when balanced.
- Regression: MAE, RMSE, sMAPE, R-squared, quantile loss.
- Forecasting: MASE, sMAPE, per horizon.
- Ranking: NDCG, MAP, MRR.
- Clustering: silhouette, Davies-Bouldin, stability across resamples, domain review.

Do
- Threshold from the cost matrix, chosen on validation.
- Calibration: reliability diagram, Brier; Platt or isotonic scaling if probabilities are
  used as probabilities.
- Slices: metric per segment and per period; name the worst slice; fairness checks where
  protected attributes exist.
- Error analysis: inspect worst residuals by hand; residual vs fitted and vs each feature;
  QQ plot; most-confused class pairs.
- Confidence: bootstrap intervals on the test metric; paired fold differences when comparing
  models, not single numbers.
- Overfit check: train vs validation gap.
- Test set scored exactly once, at the end.

## Stage 13. Explainability

- Global: permutation importance, SHAP summary, partial dependence, coefficient tables.
- Local: SHAP / LIME on individual cases.

Check
- Top features make domain sense.
- Effect directions are plausible.
- Nothing that smells like leakage sits at the top.

## Stage 14. Production readiness

- Persist the whole pipeline including preprocessing; pin versions; fix seeds.
- Score raw new rows end to end through the saved pipeline; measure latency.
- Drift monitoring: PSI / KS per feature, a performance metric to watch, a retraining trigger.
- Model card: problem, data, metrics, known limitations.

## Stage 15. Report

Problem statement, data summary, decision log with reasons, results table vs baselines,
error analysis, next steps.

---

## Cross-cutting tests (apply at every stage)

- Leakage tests: temporal availability of each feature; univariate power of each feature;
  "too good to be true" alarm when a metric is far above expectations.
- Sanity tests: shuffle labels and confirm the metric collapses to baseline; train on a
  handful of rows and confirm the model can overfit; score a duplicated row and confirm
  identical output.
- Noise floor: repeat CV with several seeds; only changes larger than the spread are real.
  (Existing skill `ml-experiment-measurement` already covers this. Point to it, do not copy.)
- Ablations: one change at a time, everything else fixed.
- Reproducibility: a clean rerun gives the same numbers.

---

## Skill boundary options (to decide after the inventory is settled)

A. One orchestrator skill + one reference file per stage. SKILL.md holds stage order,
   branching by problem type, cross-cutting tests. References hold method menus and checks.
B. Separate skills per stage (missing-values, outliers, feature-engineering, ...).
C. Hybrid: orchestrator plus 1 or 2 standalone skills for stages often requested alone
   (e.g. "just do EDA on this CSV").

Current recommendation: A, upgrade to C only if standalone use is real.
