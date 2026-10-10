# Feature engineering: method menu

Versions from PyPI's JSON API on 2026-09-27. scikit-learn classes import-checked on the installed
1.7.2 (latest release 1.9.1). "Maintained" means a release within roughly the last three years;
"quiet" means over two years without a release.

## Categorical encoders

| Method | Tool | Leak risk | Use when |
|---|---|---|---|
| One-hot | `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", min_frequency=k)` | none | Low cardinality; linear, distance, neural |
| Ordinal | `OrdinalEncoder(categories=[order])` | none | Ordered levels; trees with many levels |
| Native categorical | LightGBM `categorical_feature`, CatBoost `cat_features`, `HistGradientBoosting*(categorical_features="from_dtype")` | none | Boosted trees |
| Frequency / count | groupby size, in-fold | low | High cardinality; any family |
| Hashing | `category_encoders.HashingEncoder`, `sklearn.feature_extraction.FeatureHasher` | none | Huge or open-ended vocabularies |
| Target (cross-fitted) | `sklearn.preprocessing.TargetEncoder` (internal cross-fitting in `fit_transform`) | handled | High cardinality; default target-style encoder |
| Target variants | `category_encoders` TargetEncoder, MEstimate, JamesStein, LeaveOneOut, CatBoost, GLMM, WOE, Quantile, Summary | high unless fitted per fold | Specific smoothing needs; WOE for credit scorecards |
| Binary / base-N | `category_encoders.BinaryEncoder`, `BaseNEncoder` | none | Medium cardinality, fewer columns than one-hot |
| Dirty strings | skrub `StringEncoder`, `MinHashEncoder`, `GapEncoder`, `TableVectorizer` | none | Typos, free-form categories |
| Entity embeddings | Learned embedding layers in a neural net | none if in-fold | Neural models with high cardinality |

`category-encoders` 2.11.1 (2026-09-08); `skrub` 0.10.1 (2026-09-01).

## Numeric transforms

| Method | Tool | Notes |
|---|---|---|
| Log | `np.log1p`, `FunctionTransformer` | Positive data; interpretable |
| Box-Cox | `PowerTransformer(method="box-cox")` | Strictly positive only |
| Yeo-Johnson | `PowerTransformer()` | Any sign; default power transform |
| Quantile | `QuantileTransformer(output_distribution="normal" or "uniform")` | Robust to outliers; loses distance meaning |
| Splines | `SplineTransformer(n_knots, degree)` | Smooth non-linearity for linear models |
| Binning | `KBinsDiscretizer(strategy="quantile"/"uniform"/"kmeans")` | Step functions; loses within-bin detail |
| Optimal binning | `optbinning` 1.0.0 (2026-09-12) | Supervised, monotonic constraints, WOE and IV |
| Polynomial and interactions | `PolynomialFeatures(degree=2, interaction_only=True)` | Short lists only; feature count explodes |
| Scaling | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` (sparse) | Linear, distance and neural families only |

`feature-engine` 1.9.4 (2026-02-27) wraps many of these with column selection, plus
`CyclicalFeatures`, `DatetimeFeatures`, `RelativeFeatures`, `MathFeatures`, `DecisionTreeDiscretiser`.

## Date, time and event history

| Feature | How | Leak note |
|---|---|---|
| Calendar parts | year, month, weekday, hour, week of year | none |
| Cyclical | sin and cos of 2 pi x period position | none |
| Elapsed time | prediction moment minus signup, last purchase, last contact | Use the prediction moment, not "today" |
| Holidays | `holidays` 0.105 (2026-09-21) country and subdivision calendars | Known in advance |
| Lags and rolling windows | per entity, window ending at the prediction moment, with a gap | Must stop at the prediction moment |
| Series features | `tsfresh` 0.21.2 (2026-05-31) extracts hundreds per entity series, with relevance filtering | Compute-heavy; extract per training window |

## Relational and automated generation

| Tool | pip | Latest (released) | Maintained? | What it does |
|---|---|---|---|---|
| featuretools | `featuretools` | 1.31.0 (2024-05-14) | quiet | Deep Feature Synthesis over related tables with cutoff times |
| OpenFE | `openfe` | 0.0.12 (2023-05-25) | quiet | Generates and screens operator-based candidate features |
| AutoFeat | `autofeat` | 2.1.3 (2024-07-04) | quiet | Non-linear feature synthesis with selection, for linear models |
| featurewiz | `featurewiz` | 0.6.1 (2025-02-19) | yes | Generation plus SULOV/XGBoost-based selection |
| CAAFE | `caafe` | 0.1.6 (2023-06-17) | quiet | LLM proposes features from dataset descriptions, verified by CV |

## Feature selection

| Method | Tool | Latest (released) |
|---|---|---|
| Variance, univariate filters | `VarianceThreshold`, `SelectKBest` with `mutual_info_*`, `f_classif`, `chi2` | scikit-learn |
| mRMR | `mrmr-selection` | 0.2.8 (2023-06-30) |
| Relief family | `skrebate` | 0.8.4 (2026-09-12) |
| RFE / RFECV / sequential | `RFE`, `RFECV`, `SequentialFeatureSelector` | scikit-learn |
| SHAP-based recursive elimination | probatus `ShapRFECV` | `probatus` 3.1.5 (2026-09-08) |
| Boruta | `boruta` (BorutaPy) | 0.4.3 (2024-08-13) |
| All-relevant feature selection | `arfs` (Leshy, BoostAGroota, GrootCV) | 3.0.0 (2025-04-02) |
| Correlation-based | feature-engine `DropCorrelatedFeatures`, `SmartCorrelatedSelection` | 1.9.4 |
| Noise probe | feature-engine `ProbeFeatureSelection` | 1.9.4 |
| Permutation importance | `sklearn.inspection.permutation_importance` | scikit-learn |
| SHAP values | `shap` | 0.52.0 (2026-05-28) |
| Multicollinearity | `statsmodels.stats.outliers_influence.variance_inflation_factor` | statsmodels 0.15.0 |

Selection bias reference: Ambroise and McLachlan (2002) showed that selecting genes on the full data
before cross-validation produced near-zero error estimates on data with no signal; the fix is to
repeat selection inside every fold.
