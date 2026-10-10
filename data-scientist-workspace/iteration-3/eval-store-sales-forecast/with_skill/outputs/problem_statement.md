# Problem statement: Store sales 8-week-ahead forecast

Date: 2026-09-27   Author: data-scientist skill (automated run)   Status of assumptions: 3 ASSUMED lines open

| Field | Value | Source |
|---|---|---|
| Unit of analysis | One row = one store x one ISO week (store_id, week_start); 20 stores x ~104 weeks | data |
| Target | `sales`, weekly revenue for that store-week, as recorded | data |
| Prediction moment | End of the most recently completed week; forecast the next 8 calendar weeks per store | ASSUMED (user said "8-week-ahead forecast per store", no exact origin date given -> origin = last date in the file, 2024-12-23) |
| Decision fed | Inventory planning per store: how much stock to position for the next 8 weeks | user |
| Problem type | forecasting -> path: forecasting (`ds-forecasting`) | user |
| Primary metric | MASE (scale-free, comparable and averageable across 20 stores of very different size) because inventory planning cares about relative error per store, not one store's units dominating an average | ASSUMED (metric not specified by user; WAPE reported as the business-friendly secondary) |
| Secondary metrics | WAPE (volume-weighted, business-readable), sMAPE, per-horizon and per-store breakdowns, empirical coverage would be needed for interval-based reorder points (not built this run; noted in next steps) | this skill |
| Baseline to beat | Seasonal-naive (repeat the same ISO week from last year) and naive-last (repeat last observed week) | this skill (no existing rule/process was described by the user) |
| Success threshold | ASSUMED: beat both naive baselines by more than the noise floor on the frozen test set, since the user gave no numeric target | ASSUMED |
| Constraints | No GPU, no package installs, Windows/CPython 3.13, ~25 minute wall-clock budget for the whole project (compact mode) | user/environment |
| Data size | 2081 raw rows (2077 after de-duplication), 20 series, ~104 weeks each (2023-01-02 to 2024-12-23), 2 known covariates (promo, holiday_week) + 1 excluded (inventory_end, same-period leak) | data |
| Time and compute | ~25 minutes total wall-clock, CPU only, pandas/numpy/lightgbm/statsforecast available, no installs permitted | user |

Needs ML at all? Yes for the primary model (LightGBM on lags + known covariates beat the naive
baselines by a margin that held up on the frozen test set), but a plain seasonal-naive or
statistical model (AutoTheta) is a legitimate, much simpler fallback if interpretability or
retraining simplicity matters more than the last few points of accuracy (see decision log, Stage 8
-- the backtest could not distinguish them beyond noise; the test-set gap only appeared because the
target 8 weeks happen to be holiday-heavy).

## Open ASSUMED lines (would confirm with the data/business owner if available)
1. Forecast origin = the last date present in the file (2024-12-23); the "future" 8 weeks are
   2024-12-30 through 2025-02-17. If the business's real "today" is different, the model must be
   refit with data up to the true origin.
2. Primary metric = MASE, success threshold = "beats naive baselines beyond noise on test". No
   explicit accuracy target or cost-of-error was given.
3. Future `promo` is set to 0 (no promotion) for the delivered forecast, since no promotion
   calendar was supplied. Historical promo uplift is a measured +23% median per store (see
   decision_log.md Stage 6) -- offer this as a scenario multiplier if a promo calendar exists.
