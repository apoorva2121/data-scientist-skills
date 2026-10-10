---
name: ds-model-selection
description: Use when choosing which model families to try for a tabular classification or regression problem - linear, GLM, GAM, EBM, random forest, gradient boosting (XGBoost, LightGBM, CatBoost), tabular neural networks (RealMLP, TabM), or tabular foundation models (TabPFN, TabICL, LimiX, Mitra); when setting baselines; when the target is imbalanced, rare, skewed, a count, ordinal, multilabel, survival or uplift; when tuning a decision threshold; or when deciding whether a problem is learnable at all.
---

# Model Selection

## Overview

Stage 7 answers three questions, in order:
1. **What do simple baselines score?** Everything later must beat them by more than the noise.
2. **Which families are worth carrying into the joint search?** Screened with sensible defaults
   under the same folds, never assumed from a leaderboard.
3. **Is the problem learnable with this data?** If the best family barely beats the baseline and
   the learning curve is flat, more modelling will not help.

Imbalance handling and target transforms are chosen here too, because they depend on the family.
Hyperparameter tuning happens later, in the Stage 8 joint search (`ds-method-comparison`).

## Step 1: Baselines (always, all of them)

| Baseline | Classification | Regression | Why |
|---|---|---|---|
| Trivial | `DummyClassifier(strategy="prior")` | `DummyRegressor(strategy="median")` | The floor any model must clear |
| Business rule | The current process, or a two-line rule from Stage 3 EDA | same | What the model must beat to be worth deploying |
| Simple model | Regularised logistic regression | Ridge | Often surprisingly strong on small or mostly linear signals |
| Shallow tree | `DecisionTreeClassifier(max_depth=3)` | `DecisionTreeRegressor(max_depth=3)` | Shows whether a few thresholds explain most of it |

Score all of them with the primary metric on the Stage 2 folds and put them first in the results table.
Wrap the business rule in `scripts/rule_baseline.py` (`RuleBaseline(lambda X: score)`) so it runs
through the same folds, scorers and helpers as the models; it fits nothing and sees no labels.

## Step 2: Choose families to screen

**What wins on public benchmarks today.** TabArena (tabarena.ai, leaderboard data updated
2026-09-22, read 2026-09-27) ranks untuned tabular **foundation models** first: LimiX-2,
TabPFN-3.5, Mitra-v2 and others, all ahead of the best tuned-and-ensembled neural networks
(RealMLP, TabM) and boosted trees (LightGBM, CatBoost, XGBoost). Among boosted trees at default
settings, CatBoost ranks well above LightGBM and XGBoost. Tuned linear models rank near the bottom.
The snapshot, with times and links, is in `references/methods.md`.

**A defaults screen ranks library defaults, not model families.** The screen in Step 3 fits every
family at its out-of-the-box settings, and those settings encode each library's own idea of a
sensible starting point, not the family's potential on your data. A family can sit outside one
standard error at defaults and win the Stage 8 search after tuning - this is common, not exotic.

So when a family is **cheap**, carry it past a marginal defaults deficit and let the joint search
settle it; the cost is a few extra trials. Drop a family on the defaults screen only when it is
**expensive** as well as behind, and say which of the two reasons applied. Record the timing next
to the score so that trade-off is visible: a family that ties at 65x the cost is a different
decision from one that ties at equal cost.

**A leaderboard cannot know your data.** It averages over public datasets of limited size, all
with independent rows. Structure, leakage fixes, grouping, drift, small data and a mostly linear
signal change the order. Screen; do not assume.

**Pick three to five families** from this table, by constraints first, then data size:

| Situation | Families to screen | Notes |
|---|---|---|
| Interpretability or regulation dominates | regularised linear or logistic, EBM (`interpret`), GAM (`pygam`), scorecard (`optbinning`) | Also screen one black box, to report what interpretability costs |
| Up to about 5,000 rows | TabPFN-3.5 or TabICLv2, CatBoost (default), regularised linear | Foundation models run on CPU at this size |
| 5,000-100,000 rows, GPU available | a foundation model, CatBoost, LightGBM, optionally RealMLP or TabM | Neural nets need tuning budget and a GPU |
| 5,000-100,000 rows, CPU only | CatBoost, LightGBM, regularised linear; TabICL if a timed fold is affordable | Ask the user about a GPU if a foundation model is the likely winner (compute rule) |
| 100,000-1,000,000 rows | LightGBM, CatBoost, XGBoost; a foundation model within its row limit and with a GPU | Screen on a stratified sample, confirm on full data |
| Over 1,000,000 rows | LightGBM or XGBoost (GPU optional), linear on hashed features | Beyond current foundation-model limits |
| Strict latency or tiny memory at serving | linear, small boosted trees | Foundation models carry the training set into inference |

**Before any foundation model, check three things and log them:**
- **Licence of the weights**, not the code. TabPFN code is Apache-2.0, but the TabPFN-2.5 to 3.5
  weights are non-commercial (PyPI description, checked 2026-09-27); LimiX and EXAONE-Tabular use
  custom terms. A commercial project needs a commercial licence or a different model.
- **Where the data goes.** Hosted inference (for example the TabPFN client) sends the training
  and test rows to a third party. Confirm the user permits that before using it.
- **Limits and hardware.** TabPFN-3.5 accepts up to 1,000,000 rows and 20,000 features with a
  GPU, and about 5,000 rows on CPU; TabICLv2 is strong from 300 to 100,000 rows and up to 2,000
  features. Time one fold before committing.

## Step 3: Screen, then decide learnability

Screen the chosen families with default settings on the Stage 2 folds using `screen.py` from
`ds-method-comparison`, with the Stage 4-6 reference preprocessing for each family. Record time
per fold. Then, for the best two families, draw a learning curve:

```python
from sklearn.model_selection import learning_curve
sizes, tr, va = learning_curve(pipe, X, y, cv=folds, scoring=METRIC, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0])
```

| Evidence | Verdict | Next |
|---|---|---|
| Best family beats every baseline by more than the noise floor | learnable | Carry families forward to Stage 8 |
| Validation curve still rising at full size | data-limited | More rows would help; say so in the report |
| Large gap between training and validation curves | variance-limited | Regularise, simplify, fewer features |
| Best family within noise of the simple baseline, flat curve | not learnable yet | Stop signal: recommend better features, labels or data |

## Step 4: Imbalance (classification)

In order of preference; stop at the first that meets the objective:

1. **Change nothing but the metric and the threshold.** Use PR-AUC, precision or recall at the
   action capacity, or expected cost, then choose the threshold on validation predictions. With a
   capacity limit ("call the top 10%"), the threshold is simply the score at that rank. For a cost
   or F-beta objective, `sklearn.model_selection.TunedThresholdClassifierCV` tunes it with CV.
2. **Class weights** (`class_weight="balanced"`, `scale_pos_weight`, `auto_class_weights`).
3. **Resampling inside the training fold**: `imblearn.pipeline.Pipeline` with SMOTE or a variant,
   or undersampling. Never before the split.
4. **Balanced ensembles**: `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, `RUSBoostClassifier`.
5. **Focal loss** for boosted trees or neural nets, via a custom objective.
6. **Anomaly framing** when positives are extremely rare and heterogeneous (`ds-outliers`).

Weighting and resampling distort predicted probabilities. If probabilities are used as
probabilities, recalibrate at Stage 9. Published evidence (Elor and Averbuch-Elor, 2022, "To SMOTE,
or not to SMOTE?"; van den Goorbergh et al., 2022, JAMIA) finds that for strong classifiers,
resampling rarely beats threshold tuning and often harms calibration. Screen it; do not assume it.

## Step 5: Target transforms and special targets

| Target | Candidates |
|---|---|
| Skewed positive (prices, amounts) | `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` with smearing bias correction; Gamma or Tweedie objective |
| Counts | Poisson objective (GLM `PoissonRegressor`, LightGBM `objective="poisson"`) |
| Zero-inflated amounts | Tweedie objective; or a two-part model (probability of non-zero times amount) |
| Heavy-tailed with extreme targets | Huber or absolute-error loss; quantile objective |
| Ordinal classes | Cumulative-link model (`statsmodels` `OrderedModel`); or regression plus cut-points |
| Multiclass | Native softmax objectives; macro metrics; class weights per class |
| Multilabel | `MultiOutputClassifier`, `ClassifierChain` |
| Time to event with censoring | Survival models: Cox (`lifelines`), random survival forest and gradient-boosted survival (`scikit-survival`) |
| Effect of an action (uplift) | Meta-learners (T, S, X, R, DR) in `causalml` or `econml` |
| Ranking | `LGBMRanker`, XGBoost ranking objectives, grouped by query |

Smearing (Duan, 1983): after fitting on log y, multiply exp(prediction) by the mean of
exp(residuals) on training folds; otherwise predictions estimate the median, not the mean.

## Step 6: Carry forward

Write the surviving families (within one corrected SE of the best, plus any family required by
constraints) to the `models` slot of `search_space.json`, with their search ranges, and the chosen
imbalance and target-transform options as their own slots. The joint search tunes them together
with the preprocessing choices.

## Gate (all must exist)

- baseline table (trivial, business rule, simple model, shallow tree) with the primary metric
- family screen table with mean, spread and time per fold
- learning curves for the top two families and the learnability verdict
- for any foundation model: licence, data-location and hardware checks, logged
- `search_space.json` models slot; decision-log block with every family skipped and why

## Red flags

| Thought | Reality |
|---|---|
| "Boosting always wins, skip the baseline" | On small or mostly linear signals a regularised linear model often wins |
| "The leaderboard says foundation model X is best" | Check licence, data location, size limits and hardware, then screen |
| "SMOTE the whole dataset first" | Resampling inside the training fold only, and screen it against threshold tuning |
| "Accuracy is 91%" | With 9% positives, predicting "no" for everyone scores 91% |
| "Log the target, predict, exponentiate" | That predicts the median; apply smearing for the mean |
| "Try ten more algorithms" | If the learning curve is flat and near the baseline, the data is the limit |
