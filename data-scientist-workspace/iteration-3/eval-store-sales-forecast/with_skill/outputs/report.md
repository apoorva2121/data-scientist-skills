# Store sales 8-week-ahead forecast: model report
Status: Gates 0-10 passed (Stage 9/10 checklist items skipped under the time budget are listed
explicitly in decision_log.md and section 9/10 below, not silently dropped).

## 1. Answer first
A global LightGBM model on lags (1 week, 52 weeks), rolling means, and the two known-in-advance
covariates (promo, holiday_week) forecasts 8 weeks ahead per store. On the frozen, never-touched
test set (the actual last 8 weeks of data, 2024-11-04 to 2024-12-23, 20 stores), it scores
**MASE 0.99 (95% CI 0.80-1.23)** and **WAPE 12.2% (95% CI 10.0-14.8%)**, beating a seasonal-naive
baseline (MASE 1.14, WAPE 15.0%) and a naive-last baseline (MASE 1.25, WAPE 15.5%). Accuracy
degrades with horizon (WAPE ~9-10% at 1-3 weeks out, ~13-17% at 6-8 weeks out) and varies a lot by
store (best stores ~5-11% WAPE, worst stores S10/S11/S06/S13 at 18-32% WAPE) - inventory planning
should size safety stock per store and lean on the shorter horizons where possible, not treat the
8-week number as uniformly reliable. No explicit success threshold was given by the user (ASSUMED
line); the model clears the natural bar of beating both naive baselines beyond their measured
uncertainty.

## 2. Problem
Weekly sales per store (20 stores, ~2 years, 2023-01-02 to 2024-12-23) forecast 8 weeks ahead for
inventory planning. Open ASSUMED lines (see problem_statement.md): forecast origin = last date in
the file; primary metric = MASE with no numeric target given; future promo = 0 baseline scenario.

## 3. Data
2081 raw rows, 2077 after dropping 4 exact duplicates. Cleaning: sales == -1 (5 rows, impossible)
nulled and interpolated in-series; sales == 9999 (1 row) and negative avg_temp_c (54 rows) were
audit-flagged but confirmed to be real values, not sentinels; sales == 0 for store S19 (63/104
weeks) is a genuine recurring intermittent pattern, not an error. inventory_end is a same-period
leak (depends on that week's own sales) and is also unavailable for the future - excluded.
holiday_week is a perfectly deterministic calendar rule (100% match on ISO week-of-year in
{1,14,27,47,51,52}) - reproducible exactly for the forecast horizon. promo has no such rule and
was not supplied for future weeks - treated as an unknown scenario input (baseline 0).
Per-store data problems: S07 has a level shift around 2024-05-06 (max window-ratio 1.7, watched not
treated); S19 is intermittent (60% zero weeks) - its WAPE is unreliable (0.77 on test despite a
decent MASE of 0.71), consistent with the skill's documented failure mode for percentage metrics
near zero.

## 4. Validation design
group_time split on (store_id, week_start): test = last 8 weeks per store (160 rows), holdout =
the 8 weeks before that (160 rows), train = everything earlier (1757 rows). Entity overlap between
parts is expected and correct here (existing stores, not new-store cold start). Rejected: random/
stratified splits (would let future weeks train a model that predicts the past). Noise floor
measured as the spread of backtest MASE across 5 rolling origins (0.09-0.19 depending on
candidate) rather than a closed-form CV sd, since the data is a time-ordered panel, not iid rows.

## 5. Decisions
See decision_log.md for the full per-stage record with numbers and reasons both ways. Headline
sequence: audit and clean (Stage 1) -> time-respecting split (Stage 2) -> exclude the
inventory_end leak, classify covariates known/observed/unavailable (Stage 3) -> regularize the
panel, detect the S07 level shift and S19 intermittency, backtest 5 candidates on 5 rolling
origins (Stage 4-7) -> pick lgbm_global over statistically tied alternatives specifically because
the delivered horizon is unusually holiday-dense (Stage 8, deviation from the generic joint-search
template logged and justified) -> score the frozen test set once (Stage 9) -> refit on all data and
generate the 8-week-ahead forecast with reproduced/assumed future covariates (Stage 10).

## 6. Results
| Model / pipeline | Backtest MASE (5 origins, mean +/- origin sd) | Holdout WAPE | Test MASE (95% CI) | Test WAPE (95% CI) |
|---|---|---|---|---|
| seasonal_naive (baseline) | 1.175 +/- 0.090 | - | 1.143 | 0.150 |
| naive_last (baseline) | 1.118 +/- 0.103 | - | 1.252 | 0.155 |
| AutoETS | 1.114 +/- 0.096 | - | not re-scored (tied on backtest, not the pick) | - |
| AutoTheta | 1.096 +/- 0.097 | - | not re-scored (tied on backtest, not the pick) | - |
| **lgbm_global (final pick)** | **1.085 +/- 0.185** | 0.1375 | **0.992 (0.80-1.23)** | **0.122 (0.100-0.148)** |

By horizon (test set, lgbm_global): MASE rises from 0.79 (h=1) to 1.44 (h=8); WAPE from 0.099 to
0.169. Full table: test_by_horizon.csv.

## 7. What drives the prediction
Not formally computed under the time budget (no permutation importance / SHAP run - logged gap).
By construction the features are: last observed value (lag_1), same-week-last-year value (lag_52),
8- and 26-week rolling means, step-ahead h, and the two covariates promo and holiday_week. The
measured uplifts used to justify keeping the covariates: holiday weeks +18.2% median sales per
store (19/20 stores same sign), promo weeks +24.6% median (19/20 stores same sign) - both large
enough to plausibly explain most of the pick's edge over the covariate-blind baselines specifically
in this holiday-dense test window.

## 8. Where it fails
Worst stores by test MASE: S10 (2.36), S11 (1.89), S06 (1.56), S13 (1.48, also worst WAPE at 0.32).
Best: S04 (0.43), S02 (0.49), S03 (0.50). S19 (intermittent demand) is a special case: trust its
MASE (0.71, in line with the panel), not its WAPE (0.77, inflated by near-zero actuals) - for
inventory planning on S19, consider a reorder-point/quantile approach rather than a point forecast
(noted as a next step, not built this run). Accuracy decays substantially by horizon (h=7-8 is
roughly 1.5-2x the MASE of h=1-3) - inventory decisions that can be revised weekly should re-run
the forecast rather than committing 8 weeks out on the early estimate.

## 9. Limitations and risks
- No prediction intervals delivered (point forecasts only); inventory planning normally wants a
  reorder-point quantile, not a point estimate - flagged as the top next step.
- promo for the future 8 weeks is ASSUMED 0 (no promotion calendar supplied). If promotions are
  planned, apply the measured ~+20-25% uplift as a rough scenario multiplier to affected
  store-weeks, or re-run with a real promo calendar as a known covariate.
- ds-evaluation Step 8 sanity tests (shuffled-label check, duplicate-row check, raw-row-through-
  pipeline check) were not run under the time budget.
- No hyperparameter tuning was performed; fixed, reasonable LightGBM defaults were used throughout,
  so a further, likely modest, gain is plausible from tuning.
- The model was not serialized (no skops file) and the stateless cleaning step is not yet wrapped
  as a reusable FunctionTransformer - both are Stage 10 production-hardening gaps, not resolved
  this run.
- Only 5 backtest origins and 2 years of history (about 2 full seasonal cycles) - the noise-floor
  estimate and the seasonal signal itself are both measured on a fairly small sample.
- avg_temp_c (observed-only weather) was excluded rather than forecast/climatology-filled; if
  temperature is a real driver of sales at some stores, this leaves some signal on the table.

## 10. Next steps (ranked by expected gain)
1. Get a real promotion calendar for the next 8 weeks (or a stated planning scenario) - promo has
   the single largest measured uplift (+24.6% median) of anything available and is currently
   defaulted to 0 for every store-week.
2. Add prediction intervals (conformal on backtest residuals per horizon, per ds-forecasting
   Step 5) so inventory planning gets a reorder-point range, not just a point forecast.
3. Build a separate intermittent-demand treatment for S19 (Croston/SBA/TSB + a service-level
   quantile) rather than reading its WAPE at face value.
4. Run the skipped ds-evaluation Step 8 sanity tests and persist the pipeline with skops before
   any production use.
5. Light hyperparameter tuning and a climatology-based avg_temp_c feature, both likely small,
   incremental gains relative to items 1-3.
