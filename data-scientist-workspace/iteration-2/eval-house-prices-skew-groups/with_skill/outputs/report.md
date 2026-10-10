# House pricing model: report
Status: Stages 0-10 all passed (gates below)

## 1. Answer first
A Ridge regression on log(price) - median-imputed and winsorized numerics, one-hot categoricals,
plus house_age/total_rooms/sold_year/sold_month - estimates a new listing's sale price with a test
MAE of **$13,650** (95% CI $12,490-$14,918; MAPE 12.1%), beating a comps-style "$/sqft by
neighborhood" business rule ($20,158 MAE) by about $6,500, and the trivial median baseline
($42,385) by 3x. Each estimate ships with a 90%-target split-conformal interval (mean width
$54,714); measured test coverage is 88.7%, inside the pre-registered [87%,93%] acceptance band but
slightly under nominal and a bit tight on the low side. It beats the baseline beyond the noise
floor and lands inside the coverage band, so it meets the ASSUMED success threshold, with the
coverage caveat noted above.

## 2. Problem
Pricing tool for new listings; target = sale price (modeled as log(price)); prediction moment = at
new-listing time, so `price_per_sqft` (only known after sale) is excluded. Regression, primary
metric MAE in dollars, secondary MAPE/RMSE + 90% interval coverage. See problem_statement.md for
the 6 ASSUMED lines (log-scale target, interval-shape deliverable, business-rule baseline, no
latency SLA, quarterly retrain cadence, condition's listing-time availability) - none confirmed
with a data/business owner since none was reachable this session.

## 3. Data
4320 rows, 15 raw columns, 2021-01-01 to 2024-12-30, 4000 unique properties (320 resold once).
Cleaning (Stage 1, stateless): sqft in {50,100000} (21 rows), lot_size_sqft==9999 (1 row),
year_built in {1800,2090} (22 rows) nulled as sentinels; garage spelling variants normalized
({Yes,yes,Y,YES}->Yes, {No,no,N}->No). No rows dropped. Leak verdict (Stage 3): price_per_sqft
dropped (arithmetically derived from price/sqft, R2=1.000, only computable post-sale); all 10 other
candidate features kept as legitimate pre-listing attributes. Missingness (Stage 4): lot_size_sqft
20.6%, garage 1.4%, sqft/year_built 0.5% each, all MCAR-like (no target signal, no cross-column
predictability) -> median imputation, no indicator needed. Outliers (Stage 5): sqft and
tax_assessed_value extremes are signal (lift >2x), kept; lot_size_sqft and distance_to_center_km
heavy tails winsorized at 1%/99% for the linear model only (trees invariant to it).

## 4. Validation design
group_time split: group=property_id, time=sold_date. Test = most recent 15% by date
(2024-05-11 to 2024-12-30, 648 rows); holdout = next 15% (2023-10-18 to 2024-05-10, 649 rows);
train = earliest ~70% (2021-01-01 to 2023-10-17, 3023 rows before the group-overlap fix). 72
train rows were excluded because their property_id also appears in holdout/test (conservative
"new-entity" deployment policy - a resold property's static attributes would otherwise leak into
its own later sale); final train = 2951 rows. 5-fold expanding-window CV on train. Rejected:
pure "time" (ignores the property-repeat leak), pure "group" (ignores 4-year drift), stratified/
random (rows are not independent and deployment is on a future period). Noise floor: $289.57
(fold_sd/sqrt(5) of a deterministic Ridge/log-price reference pipeline on the fixed folds).
Test-set precision: MAE 95% bootstrap CI half-width roughly $1,200-1,400 on 648 rows.

## 5. Decisions
See decision_log.md for the full per-stage record (11 blocks, Stages 0-10). Headline calls: log-
target Ridge chosen over LightGBM by the Stage 8 one-SE rule after they tied on CV and Ridge held
up better on the locked holdout (LightGBM's CV edge did not transfer - mild overfitting to the CV
folds); median imputation for all families; winsorized tails for the linear model only; one-hot
neighborhood encoding beat cross-fitted target encoding (30 levels over about 3000 rows was too
sparse for stable target-encoding estimates); engineered calendar/domain features (house_age,
total_rooms, sold_year, sold_month) gave a real, noise-floor-beating gain.

## 6. Results
| Model / pipeline | CV metric (mean +/- sd) | Locked holdout | Test (95% CI) |
|---|---|---|---|
| Trivial baseline (median) | - | - | MAE $42,385 [39,157-45,654] |
| Business rule ($/sqft x neighborhood) | - | - | MAE $20,158 [18,650-21,661] |
| Simple model (Ridge, base features) | $13,612 +/- $647 | - | - |
| LightGBM (Stage 8 best-by-mean) | $12,108 +/- $593 | $13,003 +/- $521 | not tested (not the pick) |
| Final pipeline (Ridge, alpha=0.0115) | $12,562 +/- $597 | $12,331 +/- $454 | MAE $13,650 [12,490-14,918], RMSE $20,719, MAPE 12.1% |

## 7. What drives the prediction
Permutation importance (holdout): neighborhood (about $20,068 MAE increase) and sqft (about
$18,620) dominate; distance_to_center_km, bedrooms, year_built, condition secondary (about
$2,300-3,000 each); garage, lot_size_sqft, sold_year/month, tax_assessed_value smallest (about
$190-290). tax_assessed_value's low rank despite the top Stage-3 univariate correlation (0.937)
reflects shared credit with correlated sqft/neighborhood in a linear model, not unimportance.
Directional sanity check passed (sqft +500 -> prediction rises $312,763 -> $363,789).

## 8. Where it fails
Worst test slices by MAE: top price quintile ($26,749 - expensive homes have larger dollar errors
even at similar percent error), neighborhoods N02/N03/N07/N09/N10 ($17,000-22,000, likely thinner
training data), condition=Poor ($16,460, n=39). Interval coverage (88.7%) is under the 90% nominal
target and asymmetric (8.0% of actuals below the lower bound vs 3.2% above the upper) - the
interval runs a little tight on the downside; treat it as a planning range, not a guarantee.
worst_errors.csv holds the 20 largest test residuals for manual review (not reviewed row-by-row
this session - time budget).

## 9. Limitations and risks
All 6 ASSUMED lines in problem_statement.md are unconfirmed. Shipped model trained on 2951 of 4320
rows (holdout+test withheld for calibration/evaluation). Coverage runs slightly under nominal.
KNN/iterative imputation, CatBoost (attempted, dropped for a library incompatibility - see below),
a larger Stage 8 search (60+ trials/3 seeds vs the 25/1-seed budget cut used here), and TabPFN-3.5/
TabICLv2 (recommended by data size, not installed, TabPFN weights are non-commercial-licensed) were
all identified as worthwhile but not run. condition's availability at listing time is assumed, not
confirmed.

## 10. Next steps (ranked by expected gain)
1. Confirm the 6 ASSUMED lines with the business/data owner, especially the log-target and interval
   deliverable shape, and condition's recording time.
2. Widen the Stage 8 search (60+ trials, 3 seeds, CatBoost fixed - wrap CatBoostRegressor so it
   survives sklearn.base.clone(), or fit/copy it manually) to check whether it beats Ridge/LightGBM.
3. Push conformal coverage from 88.7% toward 90%: target a slightly higher nominal level (about
   92-93%), or move to conformalized quantile regression for a data-driven asymmetric interval.
4. Manually review worst_errors.csv (20 largest residuals) for a data or feature fix.
5. Screen KNN/iterative imputation now that the simple-imputer baseline is established, in case MAR
   structure emerges with more data.
6. Once compute/licensing allow, time one fold of TabPFN-3.5/TabICLv2 as a benchmark ceiling check.
7. At the next retrain, use all 4320 rows (this test set will be "spent") with a fresh split.

## Gates passed
Stage 0 (problem framing), Stage 1 (audit+cleaning), Stage 2 (validation design), Stage 3
(leakage hunt), Stage 4 (missing values), Stage 5 (outliers), Stage 6 (features), Stage 7
(baselines+families), Stage 8 (joint search), Stage 9 (evaluation), Stage 10 (handoff): ALL
PASSED. Budget cuts were taken at Stages 4, 5, 6, 7, 8 and 9 (fewer candidates/trials/seeds than the
skills' default guidance) to fit the 25-minute budget; each cut is logged with its reason in
decision_log.md.
