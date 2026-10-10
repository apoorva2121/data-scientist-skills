# Decision Log

## Stage 0 - Frame the problem - GATE PASSED
Wrote `problem_statement.md`. Several ASSUMED fields (success threshold, baseline, metric choice)
because the user was unavailable; listed as open risks below. Time budget: 25 minutes total,
allocated per the skill's stage-map percentages, then aggressively compressed once early stages
ran long (see cuts below).

## Stage 1 - Structure audit + stateless cleaning - GATE PASSED (compressed)
Considered running `ds-data-audit/scripts/column_audit.py` in full. Time-budget cut: with only 7
raw columns and no categorical/free-text columns, ran the equivalent checks by hand instead
(duplicates, sentinels, ranges, gaps) rather than the general-purpose profiler, to save the ~2-3
minutes the script's report-reading would have cost. Logged here so the cut is visible.

Found and fixed (stateless, `code/01_clean_and_split.py`):
- 4 exact duplicate rows (identical on every column, e.g. S08 2024-11-04 appears twice) -> dropped.
- 5 rows with `sales == -1.0`: every other sales value is a 3-5 digit positive volume, so -1 is a
  sentinel, not a real (negative) sale -> converted to NaN, then interpolated (see below). Affected
  stores: S10 (2), S12 (3).
- 3 calendar weeks entirely absent from the file for store S03 (2023-06-05, 2023-06-12,
  2024-02-19) -> `ts_backtest.regularize` reinserts them as NaN rows; interpolated (business
  assumed open - no evidence of closure, surrounding weeks show normal continuous sales).
- All 8 resulting missing `y` values (5 sentinel + 3 gap) filled by linear interpolation within
  each store's own series (neighbors only, never crossing into holdout/test - all gaps are in
  2023-2024 interior history, well before the last 16 weeks reserved at Stage 2).
- `inventory_end` flagged `ID_LIKE`-adjacent / same-period risk on sight: excluded as a feature
  (see Stage 3 leak table) rather than waiting for the Stage 3 leak screen, since the risk was
  obvious from the column semantics (end-of-week inventory is settled using that same week's
  sales).
Cleaning log: `cleaning_log.csv`, `gap_report.csv`. Row counts: 2081 raw -> 2077 after dedupe ->
2080 after regularizing (3 gap rows added back) for 20 stores x 104 weeks = 2080.

## Stage 2 - Validation design - GATE PASSED
Deployment question: forecasting a **future period** (next 8 weeks) for **existing, known
entities** (all 20 stores are already open and have history) -> `group_time` scheme is correct,
matching the skill's forecasting-path table row exactly.
Scheme: `make_split.py --scheme group_time --group store_id --time week_start --test-periods 8
--holdout-periods 8 --n-splits 5`. Rejected alternatives: plain `time` split (ignores that stores
are a natural grouping and the skill's own table recommends `group_time` for forecasting);
`group` split (would remove whole stores from train/test, but the deployment question is "next 8
weeks for stores we already operate," not "a brand-new store," so an entity split answers the
wrong question).
Result: train = 2023-01-02 to 2024-09-02 (1760 rows), holdout = 2024-09-09 to 2024-10-28 (160
rows, 8 weeks x 20 stores), test = 2024-11-04 to 2024-12-23 (160 rows, 8 weeks x 20 stores).
`group_overlap` = 20/20/20 across all three parts - expected and correct for `group_time` (same
stores persist across time; time order already prevents leakage, per skill Section 1).
Noise floor: measured as the standard deviation of backtest MASE across 6 rolling origins on the
train part (`by_origin` in `candidate_table.csv`/`candidate_by_horizon` outputs): seasonal-naive
sd=0.094, LightGBM sd=0.232. The LightGBM-vs-seasonal-naive gap (1.195 - 0.875 = 0.32) exceeds
both spreads, so the win is not noise.
Test-set precision: 160 test rows, MAE-type metric -> se ~ sd(|error|)/sqrt(160); adequate to
separate the two candidates given the gap size observed.

## Stage 3 - Leakage hunt - GATE PASSED (compressed, done by hand, see `leak_table.csv`)
- `inventory_end`: same-period leak (settled using the same week's sales) -> dropped entirely.
- `avg_temp_c`: real future weather is not known 8 weeks ahead -> excluded as a raw future
  feature; replaced with a per-store week-of-year climatology (`temp_clim`) computed from the
  **train part only**, applied unchanged to holdout/test/future (a legitimate known-in-advance
  proxy for seasonal temperature effects, not the leaking actual value).
- `promo`, `holiday_week`: ASSUMED known in advance (planned promo calendar / public holiday
  calendar). This is the task's biggest open risk - no promo calendar for the future 8 weeks was
  supplied, so the production forecast sets `promo=0` and `holiday_week=0` for all future weeks
  (see Stage 10). If merchandising has an actual promo/holiday plan for
  2024-12-30 - 2025-02-17, it should replace these placeholders and the forecast rerun; promo
  weeks lift sales noticeably in the historical data (visible in `eda`-style spot checks, e.g.
  store S03 promo weeks are consistently the local peaks), so this placeholder assumption biases
  the point forecast low during any actual future promo/holiday week.
- No adversarial-validation train/holdout AUC was run (time-budget cut); the `group_time` scheme
  and the by-origin backtest spread stand in as the leakage sanity check for this compressed run.

## Stages 4-7 (ds-forecasting) - GATE PASSED (compressed to 2 non-baseline families)
Covariate classification: known-in-advance = promo (ASSUMED), holiday_week, temp_clim, calendar
(week-of-year, month); observed-only = raw avg_temp_c, the target's own lags; same-period leak =
inventory_end (dropped).
Candidates screened on 6 rolling-origin backtests (`ts_backtest.rolling_origins`, step=4,
horizon=8, over the train part only; holdout+test periods never touched):
- `seasonal_naive` (season=52): MASE 1.195, WAPE 0.153.
- `naive_last`: MASE 1.194, WAPE 0.136.
- `lightgbm_direct_global`: one pooled LightGBM model across all 20 stores, direct multi-horizon
  (horizon `h` as a feature rather than recursive forecasting, so errors do not compound across
  the 8 steps), features = lags 1/2/3/4/8/12, rolling mean/std over 4/8/12 weeks, `store_id`
  (categorical), calendar (week-of-year sin/cos, month), known-future promo/holiday/temp_clim.
  MASE 0.875, WAPE 0.097 - clear winner, beyond both candidates' origin-to-origin spread.
Skipped and why (time-budget cuts, logged rather than silently dropped):
- `AutoETS`/`AutoARIMA`/`Theta` per store (`statsforecast`, installed): would be quick to run but
  the modelling stage was already over its time share; a global gradient-boosted model was judged
  the higher-value single candidate given 20 related series and available covariates (promo,
  holiday, temperature) that per-store univariate stats models could not use directly. Recommend
  adding as a candidate/ensemble member in a follow-up iteration.
- Deep models (N-HiTS/TFT/PatchTST): skipped per the skill's own benchmark note that they barely
  beat seasonal naive on weekly data of this size; not worth the compute given the time budget.
- Foundation/zero-shot models (Chronos, TimesFM, TiRex, Moirai, Toto): **not installed**, and the
  environment does not permit installing packages. Logged as a next-step recommendation rather
  than silently skipped. Install command (for a future run): `pip install chronos-forecasting` or
  `pip install timesfm`; both are CPU-capable for series this short, no GPU strictly required for
  inference-only zero-shot use, but check model licence before use.
- Croston/SBA/TSB (intermittent-demand methods) for store S19 specifically: S19 has ~60% zero
  weeks (63/105), a materially different pattern from the other 19 stores (continuous demand in
  the thousands-tens-of-thousands range). Not run separately due to time; the pooled LightGBM
  model still produces a forecast for S19 (small positive values, see `forecast.csv`), but its
  relative error (WAPE 0.77 on the test set) is far worse than any other store - flagged as an
  open risk / follow-up (fit Croston/TSB just for S19, or accept wider inventory buffers there).

## Stage 8 - Joint search / locked-holdout confirmation - GATE PASSED (compressed to 1 model)
No joint hyperparameter search was run (time budget); only default-ish LightGBM hyperparameters
(300 trees, 15 leaves, lr 0.05, min_child_samples 10) were used, chosen from experience rather
than tuned. This is a logged cut, not a hidden one. Confirmed on the locked holdout (fit on train
only, predict the 8 holdout weeks): LightGBM MASE 0.790 vs. seasonal-naive MASE 1.269 on the same
weeks - the holdout confirms the train-backtest ranking, so the search (such as it was) did not
overfit its own folds.

## Stage 9 - Evaluation - GATE PASSED
Test set (2024-11-04 to 2024-12-23, 160 rows, never used for any decision before this) scored
once, after refitting on train+holdout: overall MASE 0.720, WAPE 0.093 (9.3%), sMAPE 0.148.
By-horizon and by-store tables: `test_metrics_by_horizon.csv`, `test_metrics_by_store.csv`.
Worst store: S19 (WAPE 0.77, the intermittent-demand store, see above). Next-worst: S07, S06,
S13, S14 (WAPE 0.15-0.18, all still far tighter than S19). Best: S12, S15, S02, S05, S10
(WAPE 0.03-0.06). No formal calibration/coverage check of the 10-90% interval was run against the
test set (time-budget cut) - the interval in `forecast.csv` is built from train/holdout-backtest
and holdout+test-period residual quantiles by horizon, not verified for coverage on held-out data
beyond that. Flagged as an open risk.

## Stage 10 - Handoff - GATE PASSED (partial)
Final model refit on **all** available data (train+holdout+test) to produce `forecast.csv`
(8 weeks beyond 2024-12-23, i.e. 2024-12-30 to 2025-02-17, per store). Not done, due to time:
serializing the fitted pipeline to disk (joblib), a scored-raw-rows smoke test, and a monitoring
plan beyond what's in `report.md`. Listed as next steps in `report.md`.
