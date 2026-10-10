---
name: ds-feature-engineering
description: Use when creating, encoding, transforming or selecting features for a tabular or time-series model - one-hot, target, ordinal, frequency or hashing encoding of categoricals, high-cardinality columns, log or power transforms, scaling, binning, splines, interactions, date and time features, lags and rolling windows, group aggregates, automated feature generation, or feature selection (filter, wrapper, embedded, Boruta, SHAP, permutation importance); or when a model plateaus and more signal must come from the data.
---

# Feature Engineering

## Overview

Features usually carry more of the achievable gain than model choice or tuning. Two failure modes
dominate: features that are not really available at prediction time (leakage), and statistics
learned on all rows (encoders, scalers, selectors fitted outside the fold). Everything here is
therefore a pipeline step fitted in-fold, and every feature has a ledger entry saying when its
inputs exist.

## Step 1: The feature ledger

Keep `feature_ledger.csv`, one row per engineered feature or group:

| feature | group | inputs | transformation | available_at_prediction | fitted_in_fold | families |
|---|---|---|---|---|---|---|
| payment_to_income | affordability | monthly_payment, annual_income | ratio | yes: both on the application form | no (stateless) | all |
| employer_te | employment | employer_name | target encoding | yes: stated at application | yes (`TargetEncoder`) | linear, gbdt |

"available_at_prediction" must say why, not just yes. Aggregates over history are computed as of
each row's prediction moment, never over the full table (`ds-data-audit` leak types).

## Step 2: Build candidates

Ideas come from three places, in order of value: the problem statement and domain (ratios, rates,
recency, counts that mirror how the outcome happens); Stage 3 EDA notes (non-linear shapes,
thresholds, interactions, skew); and automated generation (screened like anything else).

**Categorical encoding** (the full menu is in `references/methods.md`):

| Cardinality / type | Linear, distance, neural | Boosted trees |
|---|---|---|
| Low (up to about 15 levels) | one-hot (`OneHotEncoder(min_frequency=...)` groups rare levels) | native categorical, or one-hot |
| Ordered levels | ordinal codes in the right order | ordinal codes |
| High (dozens to thousands) | target encoding (`sklearn.preprocessing.TargetEncoder`, cross-fitted internally); **frequency / count encoding** (cheap, label-free, screen it every time - it is the most commonly forgotten row in this table); hashing | native categorical; target encoding; frequency |
| Very high, IDs, free-form strings | frequency; hashing; skrub `StringEncoder` / `MinHashEncoder` / `GapEncoder` | same |
| Missing as a category | its own level | its own level |

scikit-learn's `TargetEncoder` cross-fits inside `fit_transform`, so training rows never see their
own target. The `category_encoders` target-style encoders (TargetEncoder, LeaveOneOut, James-Stein,
M-estimate, WOE, GLMM, CatBoost) do not all do that: wrap them so they are fitted per fold, or they
leak.

**Never use leave-one-out target encoding.** For a row in a group with label sum S and size n,
the LOO value is `(S - y_i) / (n - 1)`: a strictly decreasing function of the row's own label. Two
consequences, both bad:

- As a *diagnostic* (all labels visible), it inflates more than the naive statistic that includes
  the own label, because subtraction leaves a clean, invertible signature where averaging leaves a
  diluted one. A tree recovers the label in one split.
- Wrapped honestly per fold (training rows get `(S - y)/(n - 1)`, scoring rows get `S/n`), it does
  **not** inflate CV. It creates a train/serve mismatch - the model learns from a quantity that
  scoring rows never carry - and in a reproduction it scored **below** having no target encoding
  at all. So it is not a hidden leak; it is a feature that hurts.

Either way there is no setting in which it is the right choice over a cross-fitted encoder. The
same reasoning applies to any "exclude self" group statistic: deviation from the group mean
computed without self, rank within group excluding self, and so on.

**Sanity check for any target-derived feature - on rows the encoder never trained on.** Take a
held-out fold, fit the encoder on the other folds, transform the held-out rows, and within each
group compare the mean encoding for positive against negative held-out rows. An honest encoder
shows **no** within-group difference there, because the held-out labels played no part in it.

Do **not** run this comparison on training rows: every out-of-fold estimate excludes the row's own
label and is therefore mildly anti-correlated with it within a group. That is the ordinary K-fold
target-encoding bias, it is present in scikit-learn's cross-fitted `TargetEncoder`, and it is
benign - the shuffled-label canary reads chance on it. A check run on training rows would flag the
recommended encoder as a leak. The canary remains the decisive test; run it after adding any
target-derived feature.

**Encoder hyperparameters are search dimensions, not defaults.** The choice of encoder is the
small decision; its shrinkage is often the large one, and leaving it at a library default is a
choice nobody logged.

| Knob | Why it matters | What to screen |
|---|---|---|
| Target-encoder smoothing | Controls how far a thin category is pulled toward the prior. `sklearn.preprocessing.TargetEncoder(smooth="auto")` is an empirical-Bayes estimate that **on a binary target can land near or below m=1**, leaving singleton categories almost unsmoothed | `smooth` in {auto, 2, 5, 20, 100} - settings can differ by more than the noise floor |
| Encoding **granularity** | A second, independent axis: encode the exact value, and also the value floored to coarser buckets, as separate features. Coarse keys pool the thin tail that smoothing would otherwise have to rescue | exact; exact + //k; a 3-rung ladder |
| Both together | Granularity and smoothing attack the same problem from different directions. Screen the better of each, then both; do not assume one covers the other | the better of each, then both |
| `min_frequency` / rare-level grouping | Decides what counts as a level at all | 1, 10, 20 |

Screen these like any other candidate. A single encoder at library defaults is one point in a
space that routinely spans more than the noise floor.

**Features from other files.** If Stage 0b listed a reference table, a lookup, a prior extract or
the source dataset a derived table was built from, every row's **relationship to that file** is a
feature, and these are cheap, label-free and frequently strong:

| Feature | Definition |
|---|---|
| membership | is this value present in the reference table at all |
| count / multiplicity | how many reference rows carry this value |
| distance to nearest | signed or absolute gap to the closest reference value (novelty) |
| rank / quantile | where this value sits in the reference distribution |
| partial match | how many columns of this row match some reference row |

None of these uses the target, so they are stateless and can be computed once, outside the fold.
Put them in the ledger with `available_at_prediction = yes, derived from a file present at
scoring time` - and check that it really will be present at scoring time.

**Numeric transforms and scaling:**

| Need | Tool | Families that benefit |
|---|---|---|
| Skewed, positive | `log1p`; Box-Cox (`PowerTransformer(method="box-cox")`) | linear, distance, neural |
| Skewed, any sign | Yeo-Johnson (`PowerTransformer()`), `QuantileTransformer(output_distribution="normal")` | linear, distance, neural |
| Non-linear shape | `SplineTransformer`; `KBinsDiscretizer`; optimal binning (`optbinning`, monotonic WOE) | linear, GLM |
| Interactions | `PolynomialFeatures(interaction_only=True)` on a short list; domain products and ratios | linear |
| Scale | `StandardScaler`; `RobustScaler` with extremes; none for trees | linear, distance, neural |

Trees are invariant to monotonic transforms of features; do not screen transforms on the tree proxy.

**Date, time and history:** calendar parts; cyclical sin and cos for hour, weekday, month; elapsed
time since signup or last event; holiday and business-day flags (`holidays` package); lags and
rolling statistics of an entity's history ending at the prediction moment, with the Stage 2 gap.

**Aggregates and relational data:** counts, sums, means, recency, frequency and monetary features
per entity, computed point-in-time. Target-based group statistics are target encoding and must be
cross-fitted. For many related tables, Deep Feature Synthesis (`featuretools`) automates this.

**Automated generation:** OpenFE, AutoFeat, featurewiz and LLM-driven proposal (CAAFE) can suggest
features. Several are no longer actively released (dates in the reference). Treat their output as
candidates: ledger entry, availability check, in-fold screening. Never accept them unscreened.

## Step 3: Selection (inside the folds)

Selecting features on all rows and then cross-validating is a classic leak: the selection has seen
the validation rows, and scores come out optimistic. Put the selector in the pipeline.

| Method | Tool | Use when | Caveat |
|---|---|---|---|
| Drop constant or near-constant | `VarianceThreshold` | always | - |
| Redundancy | correlation above 0.9; feature-engine `SmartCorrelatedSelection` | linear models, interpretability | Keep the more reliable or more available one |
| Filter by relevance | `mutual_info_classif/regression`, ANOVA F, chi2, mRMR | many features, quick | Univariate filters miss interactions |
| Embedded | L1 (`LogisticRegression(penalty="l1")`, `Lasso`), `SelectFromModel` | linear, sparse signal | Unstable with correlated features |
| Wrapper | `RFECV`, `SequentialFeatureSelector`, probatus `ShapRFECV` | moderate feature counts | Expensive; must be nested inside CV |
| All-relevant | Boruta (`boruta`), ARFS (`arfs`), noise-probe (feature-engine `ProbeFeatureSelection`) | explanation, stable sets | Keeps weakly relevant redundant features |
| Stability selection | repeat L1 or Boruta across subsamples, keep features chosen often | small data | Compute |
| Permutation importance | `sklearn.inspection.permutation_importance` on validation folds | ranking after fitting | Correlated features share and hide importance |

Tree split-count importance is biased toward high-cardinality and continuous columns; use
permutation importance or SHAP for ranking.

## Step 4: Ablation per group

Measure each feature group by removing it, on the same folds, with `ds-method-comparison`:

```python
from screen import evaluate, summarize
sets = {"all": ALL, **{f"minus_{g}": [c for c in ALL if c not in cols] for g, cols in GROUPS.items()}}
long = evaluate({name: make_pipe(cols) for name, cols in sets.items()}, X, y, folds=folds, scoring=METRIC)
print(summarize(long, noise_floor_sd=NOISE))
```

A group whose removal does not hurt beyond the noise is a candidate to drop (simpler, cheaper,
fewer leak surfaces). A group whose removal improves the score is hurting: find out why.

## Step 5: Carry forward

Per family, carry forward two or three feature sets and encodings to `search_space.json` (slots
`features` and `encoding`), simplest first, each with its proxy scores and `applies_to`.

## Gate (all must exist)

- `feature_ledger.csv` with availability reasoning for every feature
- screening tables for encodings and transforms per family, and the group ablation table
- selection done inside the folds, with the selected-set frequency across folds
- `search_space.json` updated; decision-log block with every rejected feature idea and why

## Red flags

| Thought | Reality |
|---|---|
| "Target-encode the category on the full data" | Every row has seen its own target; cross-fit or fit per fold |
| "Select the top 20 features, then cross-validate" | The selection saw the validation rows; select inside the fold |
| "Log-transform for the boosted model" | Trees do not care; screen it on the linear proxy only |
| "The generated features look great" | Check availability and leakage before believing the score |
| "Customer's average spend is a safe feature" | Average over which period? Point-in-time, or it includes the future |
| "Feature importance says X matters most" | Split-count importance is biased; use permutation or SHAP |
