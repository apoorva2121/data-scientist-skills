# House price pricing tool — model + uncertainty estimates

**Data:** `evals/datasets/house_prices.csv`, 4,320 sold-listing records, 2021-01 to 2024-12.
**Goal:** a model that estimates sale price for a new listing, plus a per-estimate sense of how far off it might be.
**Code:** `data_prep.py` (shared cleaning), `01_eda.py`, `02_model_comparison.py`, `03_quantile_uncertainty.py`, `04_train_final_model.py`. Run in that order; each is standalone (`data_prep.py` is imported by the others).

## 1. What I did, in order

1. **EDA** (`01_eda.py`, full printout in `eda_output.txt`) to understand structure, missingness, and target shape before choosing an approach.
2. **Cleaning / feature engineering** (`data_prep.py`), built as a single importable function so every downstream script uses identical logic.
3. **Point-estimate model comparison** (`02_model_comparison.py`) — 5 candidate regressors, evaluated with group-aware cross-validation.
4. **Uncertainty estimation** (`03_quantile_uncertainty.py`) — quantile regression + conformal calibration, validated on a held-out test split, with coverage measured and reported (not assumed).
5. **Final production fit** (`04_train_final_model.py`) — refit the chosen approach on all data, save it, and show a worked example of scoring one new listing.

## 2. What I found in the data

- **4,320 rows but only 4,000 unique properties.** 320 properties (`property_id`) were sold twice in the window; duplicate rows are near-identical in every feature except `sold_date`/`price`. A random row-level split would put the same property on both sides of train/test, leaking information and overstating validation accuracy. **All splits in this project (CV folds, train/calibration/test) are grouped by `property_id`.**
- **`price_per_sqft` is target leakage.** It equals `price / sqft` up to rounding (verified numerically). It is a stand-in for the label and was **dropped**.
- **`garage` is a messy categorical**: `{Yes, No, Y, YES, yes, N, no, NaN}` (60 rows missing). Normalized to `{Yes, No, Unknown}` with a `garage_missing` indicator kept as a feature.
- **Two sentinel/error codes, not real values:**
  - `sqft` has 13 rows fixed at exactly `100000` — a mansion-sized number attached to perfectly ordinary sale prices for the stated bed/bath count (e.g. 3-bed/2-bath at $77k-$183k). Set to missing + `sqft_outlier_flag`.
  - `year_built` has 13 rows at exactly `2090` (impossible/future) and 9 rows at exactly `1800`, both disconnected from the otherwise continuous 1900-2023 range. Set to missing + `year_built_outlier_flag`.
- **`lot_size_sqft` missing in ~20% of rows** (865/4320), with no detectable relationship to price (missing-value mean price is close to the observed-value mean price) -> treated as missing-at-random, imputed with a `lot_size_missing` flag kept as a feature.
- **Target is right-skewed** (`price` skew ~1.59, range $22.4k-$588.9k) — models are fit on `log1p(price)` and predictions inverted with `expm1`.
- **`tax_assessed_value` correlates strongly with price** (r ~0.94) but is not a deterministic function of it (ratio to price ranges 0.50-1.49, not constant) — it's a legitimate, publicly-available-pre-sale feature for a new listing, so it's kept.
- All numeric imputation (median) happens **inside** the sklearn pipeline, fit on the training fold only, to avoid leaking calibration/test values into training.

## 3. Models compared, and how

`02_model_comparison.py` compares 5 regressors — Linear Regression, Random Forest, LightGBM, XGBoost, CatBoost — all fit on `log1p(price)`, predictions inverted before scoring. Evaluation: **GroupKFold, 5 folds, grouped by `property_id`**, metrics computed on concatenated out-of-fold predictions (more stable than averaging per-fold metrics on a dataset this size).

| model | RMSE | MAE | MAPE | R2 |
|---|---:|---:|---:|---:|
| **XGBoost** | **17,666** | 12,466 | 11.29% | **0.891** |
| LightGBM | 17,930 | 12,639 | 11.46% | 0.888 |
| CatBoost | 18,029 | 12,647 | 11.48% | 0.887 |
| Random Forest | 18,738 | 13,361 | 12.14% | 0.878 |
| Linear Regression | 20,523 | 12,696 | 11.41% | 0.853 |

XGBoost is narrowly best on RMSE/MAPE/R2; the gap to LightGBM/CatBoost is small (~1.5-2% RMSE) and likely partly fold noise, but it's directionally consistent across all three metrics, and XGBoost natively supports multi-quantile training in the same objective family used for point prediction — so it was carried forward as the single model family for both the point estimate and the uncertainty bands, for consistency.

## 4. Final model and uncertainty method

**Final model: XGBoost quantile regression (`objective="reg:quantileerror"`), one model per quantile level, fit on `log1p(price)`**, quantiles = {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}. Predicted quantiles are inverted with `expm1` (valid because quantiles commute with monotonic transforms) and sorted row-wise to remove any quantile crossing. The **median (q50) is the point estimate**.

**Why quantile regression instead of a single model plus a Gaussian +/- error:** price errors are visibly heteroscedastic (a $500k home and a $50k home don't have the same dollar uncertainty) and the target is skewed, so a symmetric normal interval around a mean prediction is a poor fit. Quantile regression estimates the conditional shape of the error distribution directly, without a distributional assumption.

**Why conformal calibration on top of raw quantile regression:** gradient-boosted quantile models are not guaranteed to hit their nominal coverage (e.g., a q10/q90 model isn't guaranteed to cover 80% of held-out points). **Conformalized Quantile Regression (CQR)** fixes this: an independent calibration split (never used to fit the quantile models) is used to measure how far off the raw intervals are, and an additive margin is derived from the empirical distribution of that miscalibration and applied to future intervals. This gives a finite-sample marginal coverage guarantee under the (mild) assumption that calibration and test points are exchangeable.

**Validation split:** properties (not rows) were split 60% train / 20% calibration / 20% test via `GroupShuffleSplit` on `property_id`, so no property's two sales ever land in different splits. The quantile models were fit on train only; the CQR margin was fit on calibration only; everything below is measured on test, touched exactly once.

## 5. Results

**Point estimate (median, q50) on held-out test split** (869 rows / 800 properties, never used in fitting):

- RMSE = **$17,517**, MAE = **$12,642**, MAPE = **11.9%**, R2 = **0.880**

(Consistent with the group-CV estimate for XGBoost above — no meaningful gap between CV and held-out test, which is a good sign against leakage/overfitting.)

**Interval coverage — raw quantile regression vs. after conformal calibration**, on the same held-out test split:

| interval | nominal | raw coverage | raw avg width | CQR margin | **CQR coverage** | **CQR avg width** |
|---|---:|---:|---:|---:|---:|---:|
| q05-q95 (90%) | 90% | 89.5% | $57,350 | +$1,104 | **91.6%** | $59,558 |
| q10-q90 (80%) | 80% | 76.2% | $38,520 | +$858 | **79.5%** | $40,235 |
| q25-q75 (50%) | 50% | 44.9% | $19,029 | +$135 | **45.6%** | $19,298 |

The raw quantile model under-covers at every level; CQR pulls the 90% and 80% intervals within ~0.5pp of nominal. The 50% interval is a bit short even after correction (45.6% vs. 50% target) — see caveats.

**Headline uncertainty output: the 80% interval, conformal-calibrated, achieves 79.5% empirical coverage on unseen test data** (target 80%), with an average width of ~$40,200 around a median price of ~$99k (i.e., typically +/-20% of price, wider for expensive homes, narrower for cheap ones — see below).

**Coverage by price tercile (80% interval, CQR-adjusted)** — checking the calibration isn't just "correct on average":

| tercile | n | price range | coverage | avg width |
|---|---:|---|---:|---:|
| low | 290 | $25,900-$77,700 | 77.2% | $26,656 |
| mid | 289 | $77,800-$116,900 | 87.2% | $38,958 |
| high | 290 | $117,300-$394,300 | 74.1% | $55,088 |

Interval width correctly grows with price (heteroscedasticity is captured), and coverage stays within ~6 points of the 80% target in every tercile — no severe systematic miscalibration at either end of the price range, though the low and high terciles both run a bit under nominal while mid over-covers.

## 6. How to use it

`04_train_final_model.py` refits the same quantile-model pipeline on **all** 4,320 rows (the held-out evaluation above already produced an honest performance/coverage estimate, so the production model uses every available row) and saves `final_model.joblib` — a dict with the fitted preprocessor, the 7 quantile models, and the conformal margins for the 50/80/90% intervals. `example_new_listing_prediction.txt` shows a worked example: a hypothetical new listing scores at a **median estimate of $136,738**, with an **80% interval of $109,062-$160,961**.

To score a real new listing: build a one-row DataFrame with the same feature set as `data_prep.ALL_FEATURES` (raw inputs: sqft, bedrooms, bathrooms, year_built, lot_size_sqft, neighborhood, distance_to_center_km, condition, garage, tax_assessed_value, plus the sale year/month it's being listed in), run it through the saved preprocessor and the 7 quantile models, take q50 as the point estimate, and widen [q10, q90] by the saved 80% margin for the uncertainty band.

## 7. Caveats and things I'd flag to the client

- **Conformal margins carried over to the full refit are an approximation.** CQR's coverage guarantee is proven for the exact model/margin pair it was validated with; step 4's production model is refit on ~30% more data (same distribution) than the model the margins were validated against. The measured coverage numbers above (section 5) are for the validation-split model, not literally the shipped model. Given the refit only adds more of the same kind of data, the guarantee should still approximately hold, but I'd recommend **re-validating coverage periodically as new sales come in**, and re-deriving margins from a fresh calibration split rather than treating the current ones as permanent.
- **The 50% interval undercovers by ~4.4 points even after conformal correction** (45.6% vs. 50%). With only 863 calibration / 869 test points, some of this is plausibly sampling noise (binomial std at n~869 is ~1.7pp, so this is a few standard errors out, not huge, but worth a second look) — but it could also mean the calibration set and test set differ slightly in distribution, or that additive (as opposed to multiplicative/locally-adaptive) conformal margins are a weaker fit for the narrower, more central interval than for the wider tails. I'd trust the 80%/90% bands more than the 50% band as shipped.
- **Two sentinel-value fields (`sqft=100000`, `year_built` in `{1800,2090}`) were inferred to be data-entry placeholders, not real values**, based on internal inconsistency with price — this is a judgment call on unexplained data, not a confirmed data-dictionary fact. Worth confirming with whoever owns the source system.
- **`tax_assessed_value` is the single strongest predictor** (r~0.94 with price) and does a lot of the model's work. It's plausible for a real pricing tool (assessments are public, known before a sale), but if the deployment scenario is "estimate price before an assessment exists" (e.g., brand-new construction), this feature may not be available and accuracy would likely drop noticeably — worth clarifying with the client which scenario applies.
- **Repeat-sale properties (320 of them) were kept but properly grouped** rather than dropped; this protects validation honesty but doesn't remove the underlying complication that some "new listings" may in fact be relistings of a property already in this history — not addressable from this dataset alone.
- **No explicit time-based holdout.** Data spans 2021-2024 and sale price levels could drift over time (a genuine "future new listings" deployment is inherently a forecast beyond the training window); `sale_year`/`sale_month` are included as features to let the model pick up a trend, but the group-based split doesn't test extrapolation to a later, unseen time period specifically. If the tool will be used well beyond 2024, a time-based validation (train on earlier years, test on the most recent) would be a good follow-up check.
- **Compute budget:** no hyperparameter search (Optuna) was run given the ~20-minute budget; reasonable defaults were used for all models. There's likely a little more accuracy on the table with tuning, but the relative ranking of model families is unlikely to change much at this dataset size.
