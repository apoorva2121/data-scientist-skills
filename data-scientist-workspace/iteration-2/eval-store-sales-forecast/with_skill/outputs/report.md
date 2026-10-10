# Store Sales - 8-Week-Ahead Forecast

## Problem
Inventory planning needs an 8-week-ahead weekly sales forecast for each of 20 stores. See
`problem_statement.md` for the full framing (several fields ASSUMED - user unavailable).

## Data found
2,081 raw rows, 20 stores, weekly, 2023-01-02 to 2024-12-23 (~104-105 weeks/store). No column
nulls in the raw file, but:
- 5 rows had `sales == -1.0`, a missing-data sentinel (fixed: -> NaN -> interpolated).
- 4 exact duplicate rows (dropped).
- Store S03 is missing 3 calendar weeks entirely (fixed: reinserted, interpolated).
- Store S19 has ~60% zero-sales weeks (63/105) - a genuinely different, intermittent-demand
  pattern versus the other 19 stores' continuous demand. Kept as real zeros, not treated as
  missing, but it is the hardest store to forecast (see accuracy below).
- `inventory_end` is a same-period leak for a sales forecast (settled by that week's sales) and
  was excluded as a feature.
- `avg_temp_c` is real observed weather, not knowable 8 weeks ahead; replaced by a per-store
  week-of-year climatology for the forecast horizon.
- `promo` and `holiday_week` were ASSUMED known in advance (planning/merchandising calendars);
  no such calendar was supplied for the actual forecast horizon, so the production forecast uses
  `promo=0`/`holiday_week=0` placeholders for 2024-12-30 through 2025-02-17. This is the single
  biggest open risk: any real promo or holiday in that window will be under-forecast.

## Approach
Time-based `group_time` split (last 8 weeks = test, prior 8 weeks = locked holdout, rest = train;
20/20/20 stores present in all three parts, as expected for a forecast-existing-stores problem).
Screened 3 candidates on 6 rolling-origin backtests (train part only): seasonal naive, naive-last,
and a pooled LightGBM model across all 20 stores using direct multi-horizon regression (horizon
`h` as a feature, lags 1/2/3/4/8/12 weeks, rolling mean/std, store id, calendar features, and the
known-future promo/holiday/temperature-climatology covariates). LightGBM won clearly and was
confirmed on the untouched holdout before the frozen test set was scored once.

## Results (accuracy)
| Split | Model | MASE | WAPE | sMAPE |
|---|---|---|---|---|
| Backtest (6 rolling origins, train only) | seasonal naive | 1.195 | 0.153 | 0.205 |
| Backtest (6 rolling origins, train only) | naive-last | 1.194 | 0.136 | 0.191 |
| Backtest (6 rolling origins, train only) | LightGBM (winner) | 0.875 | 0.097 | 0.163 |
| Locked holdout (8 weeks, fit on train only) | seasonal naive | 1.269 | 0.166 | 0.213 |
| Locked holdout (8 weeks, fit on train only) | LightGBM | 0.790 | 0.092 | 0.188 |
| Frozen test (8 weeks, scored once, fit on train+holdout) | LightGBM | 0.720 | 0.093 | 0.148 |

Headline: expect about 9% WAPE (total-dollar-weighted error) on a typical 8-week-ahead forecast,
roughly 40% better than a seasonal-naive baseline (WAPE 0.153-0.166) measured the same way, on
both an independent rolling-origin backtest and a single frozen test set.

By horizon (test set, `test_metrics_by_horizon.csv`): error is not flat - WAPE is best at h=1-2
(about 0.07), worst at h=7 (0.128) and h=5 (0.111), with h=8 back down to 0.109. Plan larger
buffers for weeks 5-7 of the horizon than for weeks 1-2.

By store (test set, `test_metrics_by_store.csv`):
- Worst: S19, WAPE 0.77 - the intermittent-demand store; near-zero actuals make relative error
  huge even though the model tracks its occasional spikes reasonably (MASE 0.71, in line with
  other stores). Treat S19's forecast as low-confidence; consider a dedicated intermittent-demand
  method (Croston/TSB) or a wider safety-stock policy instead of trusting the point forecast.
- Next-worst: S07, S06, S13, S14 (WAPE 0.15-0.18).
- Best: S12, S15, S02, S05, S10 (WAPE 0.028-0.055).

## Backtest scheme
`group_time` split; noise floor = spread of backtest MASE across 6 rolling origins (seasonal
naive sd 0.094, LightGBM sd 0.232); the LightGBM-vs-baseline gap (about 0.32 MASE) exceeds both
spreads. Full detail in `decision_log.md` Stages 2 and 8.

## Data problems found
See "Data found" above and `leak_table.csv`, `cleaning_log.csv`, `gap_report.csv` for the full
detail: sales sentinel (-1), 4 exact duplicate rows, 3 missing calendar weeks for S03, S19's
intermittent-demand pattern, `inventory_end` same-period leak, and unknown future promo/holiday
plan.

## Limitations / open risks (ASSUMED lines to confirm with the user or data owner)
1. Success threshold and "baseline to beat" were ASSUMED (no target or current process given).
2. `promo` and `holiday_week` were ASSUMED knowable in advance; the actual forecast horizon uses
   placeholder zeros for both because no future calendar was supplied - rerun with a real
   promo/holiday plan before trusting the numbers for weeks with known promotions or holidays.
3. S19's intermittent pattern was not modelled with a dedicated method (Croston/TSB) - time
   budget. Its forecast is the least reliable of the 20 stores.
4. No formal 10-90% interval coverage check against the test set; the intervals in `forecast.csv`
   are backtest/holdout-residual-based, not independently validated.
5. Statistical family (AutoETS/AutoARIMA via `statsforecast`, installed but not run) and zero-shot
   foundation models (Chronos/TimesFM/TiRex - not installed, no GPU) were skipped for time/compute
   reasons, logged in `decision_log.md`, and are the natural next candidates to try.
6. No hyperparameter search was run for LightGBM (time budget); defaults-ish settings were used.

## Next steps
1. Get a real promo/holiday calendar for 2024-12-30 through 2025-02-17 and rerun
   `code/03_evaluate_and_forecast.py`'s final-forecast block with it.
2. Add a dedicated intermittent-demand model for S19 (and screen for any other stores that drift
   that way over time).
3. Try `statsforecast` AutoETS/AutoARIMA as an additional candidate/ensemble member; average with
   LightGBM if it backtests within the noise floor (the skill's "combine" step, not yet done).
4. Validate interval coverage on the test set; if under-covering, widen via conformal calibration.
5. Serialize the fitted pipeline (joblib) with pinned versions/seed, smoke-test on raw rows, and
   write a monitoring plan (Stage 10 was left partial due to time).

## Artifacts in this folder
`problem_statement.md`, `decision_log.md` (one block per stage), `leak_table.csv`,
`cleaning_log.csv`, `gap_report.csv`, `candidate_table.csv`, `candidate_by_horizon.csv`,
`candidate_by_series.csv`, `test_predictions.csv`, `test_metrics_by_horizon.csv`,
`test_metrics_by_store.csv`, `forecast.csv` (final 8-week-ahead forecast per store, columns:
store_id, week_start, h, forecast_sales, q10, q90), `split/` (frozen split spec),
`panel_clean.csv` (cleaned/regularized panel), and `code/01_clean_and_split.py`,
`code/02_backtest_candidates.py`, `code/03_evaluate_and_forecast.py` (rerunnable end to end).

## Time budget
Used the full ~25-minute budget; stages 1, 3, and the joint search (Stage 8) were compressed as
described in `decision_log.md` (manual checks instead of running every bundled script; 1 ML
candidate carried forward instead of 2-3; no hyperparameter search). Correctness gates
(sentinel/duplicate handling, leak table, time-based split, test-scored-once) were not cut.

## Skill feedback (things that were unclear, wrong, or hard to follow)
- `ds-forecasting/scripts/ts_backtest.py`'s `backtest()` builds a `future["h"]` column internally
  but then calls `forecaster(hist, fut[cols].copy())` without `h` in `cols` (only
  `["unique_id","ds",*known_future]`). A custom forecaster that needs to know which step-ahead
  it's predicting (any "direct" multi-horizon model, which the skill explicitly recommends) must
  recompute `h` itself via `future.groupby("unique_id").cumcount()+1` - the module's doc-string
  example only shows `seasonal_naive`/`naive_last`, which don't need `h` as an input, so this gap
  isn't obvious until you hit a `KeyError: 'h'`. Worth either passing `h` through in `cols` or
  documenting the recompute step.
- `ds-forecasting/SKILL.md` recommends MASE/WAPE and to "always report seasonal naive and naive"
  but doesn't say how to present the by-store table for a store like S19 (majority-zero weeks) -
  WAPE is still very large in relative terms even though MASE looks normal. Guidance on which
  metric to lead with per-store (vs. pooled) would help set planner expectations.
- No other gaps found; `make_split.py`'s `group_time` scheme and `ts_backtest.py`'s
  `regularize`/`rolling_origins`/`backtest`/`summarize` worked exactly as documented once the `h`
  issue above was worked around.
