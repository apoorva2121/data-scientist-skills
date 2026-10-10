# Forecasting: benchmark snapshot, methods, packages

Checked 2026-09-27. Versions from PyPI's JSON API; code licences from the GitHub API; benchmark
tables read from the fev-bench Hugging Face space (`autogluon/fev-bench`, data updated 2026-09-18).
GIFT-Eval (`Salesforce/GIFT-Eval`, updated 2026-09-23) is the other major public leaderboard.

## fev-bench snapshot: weekly series, MASE (32 models)

Skill score = percentage improvement over seasonal naive (0 by definition). Win rate = share of
pairwise comparisons won.

| Position | Model | Win rate | Skill score | Type |
|---|---|---|---|---|
| 1 | Chronos-2 | 78.7 | 31.5 | foundation, zero-shot |
| 2 | TimesFM-3 | 78.4 | 31.9 | foundation, zero-shot |
| 3 | FlowState | 72.5 | 26.9 | foundation |
| 4 | t0-beta | 71.2 | 29.8 | foundation |
| 5 | TiRex-2 | 71.2 | 29.3 | foundation |
| 6 | TimesFM-2.5 | 70.8 | 26.5 | foundation |
| 7 | TabPFN-TS-3 | 67.0 | 29.0 | tabular foundation model applied to series (slow: ~200 s per 100 series) |
| 19 | LightGBM (on lags) | 45.5 | 19.4 | ML, trained |
| 21 | Statistical ensemble | 44.0 | 15.4 | statistical |
| 23 | AutoETS | 38.5 | 11.6 | statistical |
| 25 | TFT | 32.1 | 11.5 | deep, trained |
| 26 | PatchTST | 28.8 | 9.3 | deep, trained |
| 27 | AutoARIMA | 27.9 | 3.7 | statistical |
| 29 | Seasonal naive | 25.6 | 0.0 | baseline |
| 31 | DeepAR | 17.8 | 6.6 | deep, trained |

The daily table (same source) has t0-beta, TimesFM-3, Chronos-2 and TiRex-2 on top. Benchmarks
cover public univariate-heavy series; covariate-driven problems (promotions, prices) can reorder
them. Re-read the leaderboard at the start of each project.

## Foundation models

| Model | pip | Latest (released) | Code licence (GitHub) | Notes |
|---|---|---|---|---|
| Chronos-2, Chronos-Bolt | `chronos-forecasting` | 2.3.2 (2026-09-08) | Apache-2.0 | Amazon; check model card for weights and covariate support |
| TimesFM | `timesfm` | 3.0.2 (2026-09-09) | Apache-2.0 | Google |
| TiRex | `tirex-ts` | 1.4.2 (2026-06-09) | custom (NOASSERTION) | NXAI; read the licence before commercial use |
| Moirai-2.0 | `uni2ts` | 2.0.0 (2025-11-04) | Apache-2.0 | Salesforce; weights licence on the model card |
| Toto | `toto-ts` | 0.2.0 (2026-02-26) | Apache-2.0 | Datadog |
| TabICLForecaster | `tabicl[forecast]` | 2.2.0 (2026-09-02) | BSD-3-Clause | Zero-shot forecasting from a tabular in-context model |
| TimeGPT (API) | `nixtla` | 0.9.0 (2026-09-21) | client | Hosted: data leaves the machine; needs an API key and user permission |

## Libraries

| Library | pip | Latest (released) | Covers |
|---|---|---|---|
| StatsForecast | `statsforecast` | 2.1.1 (2026-07-16) | AutoETS, AutoARIMA, AutoTheta, MSTL, TBATS, Croston family, ADIDA, IMAPA, conformal intervals, fast cross-validation |
| MLForecast | `mlforecast` | 1.1.0 (2026-07-10) | Global LightGBM/XGBoost with lag and rolling features, direct and recursive, conformal intervals |
| NeuralForecast | `neuralforecast` | 3.2.2 (2026-09-08) | N-HiTS, N-BEATS, TFT, PatchTST, TiDE, DeepAR and more |
| HierarchicalForecast | `hierarchicalforecast` | 1.5.3 (2026-09-24) | BottomUp, TopDown, MiddleOut, MinTrace, ERM reconciliation |
| utilsforecast | `utilsforecast` | 0.2.16 (2026-04-27) | Losses (MASE, RMSSE, scaled CRPS, quantile), evaluation, plotting |
| sktime | `sktime` | 1.2.0 (2026-09-22) | Unified API over many forecasters, reduction, pipelines, evaluation |
| Darts | `darts` | 0.47.0 (2026-09-04) | Statistical, ML and deep models, backtesting, covariates |
| GluonTS | `gluonts` | 0.17.0 (2026-07-31) | Probabilistic deep models (DeepAR etc.), evaluation |
| skforecast | `skforecast` | 0.25.0 (2026-09-11) | Recursive and direct forecasting with any scikit-learn regressor |
| Prophet | `prophet` | 1.4.0 (2026-08-15) | Trend + seasonality + holidays; interpretable; often beaten on benchmarks |
| pmdarima | `pmdarima` | 2.1.1 (2025-11-17) | auto_arima |
| statsmodels | `statsmodels` | 0.15.0 (2026-08-27) | SARIMAX, ETS, STL, unobserved components |
| ruptures | `ruptures` | 1.1.10 (2025-09-10) | Change-point detection |
| tsfresh | `tsfresh` | 0.21.2 (2026-05-31) | Feature extraction per series window |
| MAPIE | `mapie` | 1.5.0 (2026-08-05) | Conformal intervals, including time-series variants |

## Metrics

| Metric | Formula (per series, then averaged) | Use |
|---|---|---|
| MASE | MAE / in-sample MAE of the seasonal naive | Default scale-free point metric |
| RMSSE | RMSE analogue of MASE | When large errors matter more (M5 used it) |
| WAPE | sum abs error / sum abs actual | Volume-weighted, business-friendly |
| sMAPE | mean 2 abs(e) / (abs(y) + abs(yhat)) | Legacy comparisons; unstable near zero |
| Pinball / quantile loss | per quantile | Probabilistic forecasts |
| CRPS, weighted quantile loss (WQL) | integral of quantile losses | Whole predictive distribution |
| Coverage | share of actuals inside the interval | Calibration of intervals |

## Validation vocabulary

Rolling origin (expanding window) versus sliding window (fixed-length training); step between
origins; horizon H; gap between training end and first forecast period when features look back;
refit at every origin versus fit once. Nixtla's `cross_validation` and sktime's
`ExpandingWindowSplitter` / `SlidingWindowSplitter` implement the same designs as `scripts/ts_backtest.py`.
