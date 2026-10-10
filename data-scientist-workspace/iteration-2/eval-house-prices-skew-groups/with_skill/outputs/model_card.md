# Model card: house pricing tool

## Intended use
Estimate a sale price (point estimate + 90%-target interval) for a NEW listing at the moment it
enters the pricing tool, to anchor a suggested list price. Not intended for: valuing already-sold
properties for tax/legal purposes, or as the sole basis for a binding price without agent review
(see Limitations - interval width and worst slices below).

## Data
4320 historical sales, 2021-01-01 to 2024-12-30, 4000 unique properties (320 resold once within the
window). 9 features used: sqft, bedrooms, bathrooms, year_built, lot_size_sqft, distance_to_center_km,
neighborhood (30 levels), condition (5 ordered levels), garage (yes/no), tax_assessed_value, plus 4
engineered: house_age, total_rooms, sold_year, sold_month. `price_per_sqft` excluded (derived from
the target, only knowable after sale - see decision_log.md Stage 3).

## Pipeline
`clean_raw` (sentinel->null, spelling normalization) -> calendar parts -> house_age/total_rooms ->
winsorize(1%/99%) lot_size_sqft & distance_to_center_km -> median-impute numerics, standardize ->
most-frequent-impute + one-hot categoricals -> Ridge(alpha=0.0115) on log(price). Chosen over
LightGBM by the Stage 8 1-SE rule (statistically tied on CV, Ridge held up better on the locked
holdout - see decision_log.md Stage 8).

## Metrics (test set, 648 rows, 2024-05-11 to 2024-12-30, scored once)
| | MAE | RMSE | MAPE |
|---|---|---|---|
| Trivial (median) | $42,385 [39,157-45,654] | $60,414 | 37.2% |
| Business rule ($/sqft x neighborhood) | $20,158 [18,650-21,661] | $27,994 | 16.5% |
| **Final pipeline** | **$13,650 [12,490-14,918]** | **$20,719** | **12.1%** |

CV mean MAE $12,562, holdout MAE $12,331 - test is consistent with both (no leak red flag), though
on the higher side, plausibly because 2024 H2 was a somewhat harder pricing period (see slices).

## Uncertainty
Split-conformal (hand-implemented, MAPIE not installed), calibrated on the 649-row holdout,
nonconformity = |log(y)-log(pred)|, target 90% coverage. **Measured test coverage: 88.7%**
(within the pre-registered [87%,93%] acceptance band, but under nominal and asymmetric: 8.0% of
actual prices fell below the interval vs 3.2% above - the interval runs a bit tight on the low
side). Mean interval width $54,714 (about half the test median price) - treat as a quoting range,
not a precise figure.

## Where it works less well
- Top price quintile: MAE $26,749 (expensive homes are harder to price exactly, in dollar terms).
- Smaller neighborhoods (N02, N03, N07, N09, N10): MAE $17,000-22,000, likely thinner training data.
- `condition = Poor` (n=39 in test): MAE $16,460.
- No protected-attribute columns present in this dataset; no fairness table produced.

## What drives the prediction
Permutation importance (holdout): neighborhood and sqft dominate (~$20k and ~$19k MAE increase
when permuted); distance_to_center_km, bedrooms, year_built, condition are secondary (~$2-3k each).
tax_assessed_value ranks lowest despite the highest single-column correlation with price (Stage 3),
because it is collinear with sqft/neighborhood and a linear model splits credit among correlated
inputs - not evidence it is unimportant, just redundant here.

## Limitations, risks, ASSUMED lines
- Log-scale target modeling, the 90% interval target, the business-rule baseline, quarterly
  retraining, and `condition`'s listing-time availability are all ASSUMED (problem_statement.md);
  none were confirmed with a data/business owner because none was reachable this session.
- Coverage is measured slightly under nominal and asymmetric (see Uncertainty); do not treat the
  interval as a hard guarantee, especially on the low side.
- Trained on 2951 rows (train-only, holdout/test withheld for calibration/evaluation); a future
  retrain can use all 4320 rows once this test set is "spent".
- KNN/iterative imputation, CatBoost, a larger Optuna search (60+ trials/3 seeds), TabPFN/TabICL
  foundation models, and manual review of the 20 worst test errors were all identified as
  worthwhile but not completed under the 25-minute time budget (see report.md Next steps).
- CatBoost was attempted and dropped due to a confirmed library incompatibility
  (`CatBoostRegressor(cat_features=...)` breaks `sklearn.base.clone()`), not a modeling judgment.

## Monitoring plan
- **Input drift**: per-feature PSI/KS on sqft, tax_assessed_value, neighborhood mix, monthly, vs
  the training distribution; alert if PSI > 0.2 on any top-5-importance feature.
- **Prediction drift**: monthly mean/median predicted price vs. the training-period baseline;
  large shifts may mean market movement the model (fit through 2023-10-17 for point estimates, or
  through 2024-12-30 for interval calibration) has not seen.
- **Delayed-label tracking**: once a listing's actual sale price is known, log realized MAE/coverage
  in rolling 90-day windows; retrain if realized MAE exceeds the test MAE by more than ~2x the
  bootstrap CI width, or if realized coverage drifts outside [80%, 95%].
- **Data validation**: re-run column_audit.csv's sentinel/range rules on every incoming batch
  (sqft in {50,100000}, lot_size_sqft==9999, year_built in {1800,2090}, garage spelling) before
  scoring; reject or flag rows that fail.
- **Retraining trigger**: quarterly (ASSUMED cadence) or on realized-performance drift above,
  whichever comes first; always with a fresh chronological test set, never reusing this one.
