# Model selection: families, benchmark snapshot, packages

Checked 2026-09-27. Versions from PyPI's JSON API; licences from PyPI metadata, package
descriptions or the GitHub API as stated; class names import-checked locally (scikit-learn 1.7.2,
lightgbm 4.7.0, imbalanced-learn 0.14.0, statsmodels, catboost).

## TabArena snapshot (tabarena.ai, leaderboard data updated 2026-09-22)

All tasks, all datasets, no imputation. 82 entries: 28 foundation models, 27 tree-based,
15 neural networks, 6 baselines, 6 other. Elo is relative (RandomForest default = 1000).
Median times are per 1,000 training rows on TabArena's hardware (foundation models on GPU).

| Position | Model | Type | Elo | Train s/1K | Predict s/1K |
|---|---|---|---|---|---|
| 1 | LimiX-2 (default) | foundation | 1966 | 30.9 | 9.03 |
| 2 | TabPFN-3.5 (default) | foundation | 1888 | 1.84 | 0.47 |
| 6 | Mitra-v2 (default) | foundation | 1784 | 57.1 | 0.88 |
| 11 | TabICLv2 (default) | foundation | 1581 | 0.75 | 0.14 |
| 16 | RealMLP (tuned + ensembled) | neural | 1498 | 2951 | 12.0 |
| 19 | TabM (tuned + ensembled) | neural | 1444 | 2450 | 2.25 |
| 20 | LightGBM (tuned + ensembled) | tree | 1430 | 417 | 2.64 |
| 22 | CatBoost (tuned + ensembled) | tree | 1419 | 1346 | 0.34 |
| 30 | CatBoost (default) | tree | 1378 | 5.88 | 0.03 |
| 31 | XGBoost (tuned + ensembled) | tree | 1373 | 693 | 1.69 |
| 47 | EBM (tuned + ensembled) | tree (interpretable) | 1270 | 1711 | 0.14 |
| 51 | XGBoost (default) | tree | 1212 | 1.94 | 0.12 |
| 57 | LightGBM (default) | tree | 1187 | 1.96 | 0.14 |
| 74 | RandomForest (default) | tree | 1000 | 0.43 | 0.05 |
| 79 | Linear (tuned) | baseline | 935 | 238 | 0.09 |

Other top-ten foundation entries: Causilo, TabPFN-3.5-Fast, TabFM (Google), EXAONE-Tabular,
TabPFN-3, Xiaomi-TabLDM. Subsets for small, medium, binary, imbalanced and other slices are in the
space's `data/` folder (Hugging Face space `TabArena/leaderboard`); read the slice closest to the
project before choosing. Re-read the leaderboard at the start of each project: it moves monthly.

## Tabular foundation models

| Model | Package | Latest (released) | Code licence | Weights licence | Limits and hardware (from the package description) |
|---|---|---|---|---|---|
| TabPFN-3.5 / 3.5-Fast | `tabpfn` | 9.0.0 (2026-09-15) | Apache-2.0 | Non-commercial (TabPFN-2.5, 2.6, 3, 3.5); commercial licence from Prior Labs; login to accept terms | Up to 1,000,000 rows and 20,000 features; GPU recommended (8-16 GB); CPU about 5,000 rows |
| TabICLv2 | `tabicl` | 2.2.0 (2026-09-02) | BSD-3-Clause (PyPI metadata) | open, including pre-training | Strong at 300-100,000 rows, up to 2,000 features; larger with offloading; also `TabICLForecaster` |
| TabDPT | `tabdpt` | 1.3.1 (2026-09-21) | not checked | not checked | see package |
| Mitra / Mitra-v2 | via `autogluon` | 1.6.3 (2026-09-18) | Apache-2.0 (repo) | check model card | via AutoGluon tabular |
| LimiX-2 | see arXiv 2609.17488 | - | custom (GitHub: NOASSERTION) | check | see paper |
| TabFM | github.com/google-research/tabfm | - | Apache-2.0 (repo) | check | see repo |
| Causilo | github.com/nums-ai/causilo | - | Apache-2.0 (repo) | check | see repo |
| EXAONE-Tabular | github.com/LGAI-Research/EXAONE-Tabular | - | custom (GitHub: NOASSERTION) | check | see repo |

Code licence and weights licence are separate. Read the model card's weights licence before any
commercial use, and never send data to a hosted endpoint without the user's permission.

## Boosted trees

| Library | pip | Latest (released) | Strengths |
|---|---|---|---|
| CatBoost | `catboost` | 1.2.10 (2026-02-18) | Best default among boosted trees on TabArena; ordered target statistics for categoricals; `auto_class_weights` |
| LightGBM | `lightgbm` | 4.7.0 (2026-07-18) | Fast on large data; native categorical and NaN; `objective` poisson, tweedie, gamma, quantile, huber, lambdarank |
| XGBoost | `xgboost` | 3.4.1 (2026-08-15) | Mature GPU training; ranking and survival objectives |
| scikit-learn HistGradientBoosting | `scikit-learn` | 1.9.1 latest (1.7.2 local) | No extra dependency; native NaN and categorical |

All three external libraries support monotonic constraints, which matter where regulators or
domain logic require a feature to only raise or only lower the prediction.

## Neural networks for tables

| Model | pip | Latest (released) | Notes |
|---|---|---|---|
| RealMLP (and tuned defaults for GBDTs) | `pytabkit` | 1.7.3 (2026-01-06) | Strong tuned MLP; needs tuning budget and GPU |
| TabM | `tabm` | 0.0.3 (2025-08-14) | Parameter-efficient ensemble MLP |
| FT-Transformer, ResNet, MLP | `rtdl-revisiting-models` | 0.0.2 (2023-11-30) | Reference implementations; quiet |
| Assorted (TabNet, GANDALF, FT-Transformer, ...) | `pytorch-tabular` | 1.2.0 (2026-01-26) | Framework for many tabular nets |

## Interpretable models

| Model | pip | Latest (released) |
|---|---|---|
| Explainable Boosting Machine (EBM) | `interpret` | 0.7.8 (2026-03-17) |
| Generalised additive models | `pygam` | 0.12.0 (2025-12-18) |
| GLMs with inference | `statsmodels` | 0.15.0 (2026-08-27) |
| Scorecards (binning, WOE, logistic) | `optbinning` | 1.0.0 (2026-09-12) |

## Imbalance tools

`imbalanced-learn` 0.14.2 (2026-06-07; 0.14.0 installed locally): over-samplers
(`RandomOverSampler`, `SMOTE`, `SMOTENC` for mixed types, `SMOTEN` for categorical, `BorderlineSMOTE`,
`SVMSMOTE`, `KMeansSMOTE`, `ADASYN`), under-samplers (`RandomUnderSampler`, `TomekLinks`,
`EditedNearestNeighbours`, `NearMiss`, `CondensedNearestNeighbour`, `OneSidedSelection`,
`NeighbourhoodCleaningRule`, `InstanceHardnessThreshold`), combined (`SMOTEENN`, `SMOTETomek`),
ensembles (`BalancedRandomForestClassifier`, `BalancedBaggingClassifier`, `EasyEnsembleClassifier`,
`RUSBoostClassifier`), and `imblearn.pipeline.Pipeline`, which applies samplers during `fit` only.

Threshold tools in scikit-learn (1.7.2): `TunedThresholdClassifierCV` (tunes the cut-off for a
metric with CV) and `FixedThresholdClassifier`.

## Special targets

| Target | Package | Latest (released) |
|---|---|---|
| Survival | `lifelines` | 0.30.3 (2026-03-05) |
| Survival (ML) | `scikit-survival` | 0.28.0 (2026-07-05) |
| Uplift / heterogeneous treatment effects | `causalml` | 0.17.0 (2026-07-04) |
| Uplift / double ML | `econml` | 0.17.0 (2026-07-31) |
| Probabilistic regression | `ngboost` | 0.5.11 (2026-06-26) |
| Ordinal | `statsmodels.miscmodels.ordinal_model.OrderedModel` | statsmodels 0.15.0 |
