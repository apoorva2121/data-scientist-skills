# Problem Statement

| Field | Value |
|---|---|
| Unit of analysis | One (store, ISO week) pair. 20 stores, weekly sales, ~2023-01-02 to 2024-12-23 (~104-105 weeks/store). |
| Target | `sales`: total weekly sales for the store, in currency units. Value `-1.0` found in 5 rows is a missing-data sentinel, not a real value (converted to NaN, then interpolated - see decision log). |
| Prediction moment | End of the most recently completed sales week (the forecast is produced "as of" the last observed Monday-week-start row per store). |
| Decision fed | Inventory planning per store: how much stock to have on hand to cover demand over the next 8 weeks. |
| Problem type | Forecasting (multi-series, horizon H=8 weeks, direct multi-horizon regression). |
| Primary metric + reason | MASE (scale-free, comparable and averageable across 20 stores whose sales levels differ 5-10x) and WAPE (business-friendly: total $ forecast error / total $ actual, volume-weighted so large stores are not swamped by small-store noise). ASSUMED: avoid MAPE - store S19 has ~60% zero-sales weeks, which makes MAPE explode/undefined. |
| Secondary metrics | sMAPE (cross-check), by-horizon breakdown (h=1..8), by-store breakdown (worst/best stores), empirical 10-90% interval coverage from backtest residuals. |
| Baseline to beat | Seasonal naive (same week last year) and naive-last (last observed week), both backtested on the same rolling origins. ASSUMED: no current manual/Excel forecasting process was described, so these statistical baselines stand in for "current process." |
| Success threshold | ASSUMED (no target given): beat both baselines by more than the noise floor (origin-to-origin MASE spread) on the backtest, and land in a WAPE range useful for inventory buffers (single-digit-to-low-teens %). Achieved: LightGBM WAPE ≈ 9% on the frozen test vs. seasonal-naive ≈ 15%. |
| Constraints | No GPU, CPU-only laptop; no package installs permitted (no time-series foundation model such as Chronos/TimesFM/TiRex is installed, so the zero-shot family is skipped and logged, not silently dropped); 25-minute wall-clock budget for the whole engagement. |
| Data size | 2,081 raw rows, 20 stores, ~104-105 weeks/store, 5 columns of covariates + target. Small by ML standards, adequate for a handful of lag/seasonal features per store plus a pooled (global) model across stores. |
| Time and compute budget | 25 minutes total. Actual stage split (compressed heavily, see decision log): ~35% framing+audit+split, ~10% leakage/EDA, ~40% modelling+backtest, ~15% evaluation+writeup. Each candidate model backtest ran in well under a minute. |

**Does this need ML at all?** A pure seasonal-naive rule (same week last year) is a legitimate, cheap baseline and is reported alongside the model. It is measurably worse (test-comparable backtest MASE 1.20 vs. LightGBM 0.87, holdout MASE 1.27 vs. 0.79), so a fitted model is justified here, but seasonal naive remains the fallback if the pipeline cannot be maintained.
