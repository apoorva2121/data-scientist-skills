---
name: ds-forecasting
description: Use when the target is a future value of one or many time series - demand, sales, traffic, load, prices, inventory, capacity - at a stated horizon; when choosing between seasonal naive, ETS, ARIMA, Theta, Prophet, LightGBM on lags, deep models (N-HiTS, TFT, PatchTST) or zero-shot foundation models (Chronos, TimesFM, TiRex, Moirai, Toto); when backtesting a forecast; when series have gaps, intermittent zeros, level shifts or promotions; or when forecasts must add up across a hierarchy.
---

# Forecasting

## Overview

Forecasting replaces Stages 4-7 of the `data-scientist` workflow when the target is a future value
of a time-indexed series. Stages 0-3 still run first (framing, audit, a time-based split with
`--test-periods H`, leak hunt), and Stages 8-9 follow (`ds-method-comparison`, `ds-evaluation`).

Three things make forecasting different:
- **Only the past is available.** Every feature at forecast time must be known at the origin, or
  known in advance for the future periods (calendar, planned promotions). Anything else leaks.
- **Accuracy depends on the horizon.** Report it per step ahead, not only on average.
- **Seasonal naive is a strong, honest baseline.** On the current public weekly benchmark, trained
  deep models (TFT, PatchTST, DeepAR) barely beat it, while zero-shot foundation models beat it by
  about 30% (numbers in `references/methods.md`).

## Step 1: Regularise and inspect the panel

```python
import sys; sys.path.insert(0, "<path to ds-forecasting/scripts>")
from ts_backtest import regularize, rolling_origins, backtest, seasonal_naive, naive_last, summarize
panel, gaps = regularize(df, id_col="<series id>", time_col="<date>", target="<y>", freq="W-MON")
```

`gaps` separates **periods absent from the data** from **rows present with y missing** (often a
nulled sentinel). Then check every series for level shifts and intermittency:

```python
from ts_backtest import break_report
breaks = break_report(panel, window=13, ratio=1.3, season=52)   # season = periods per cycle
```

It compares each window with the same weeks one season earlier, so seasonality does not read as a
shift. Decide per series and log it:

| Finding | Treatment |
|---|---|
| Period absent, business was open | Missing: interpolate within training history, or leave NaN for a NaN-aware model |
| Period absent, nothing happened | A true zero: fill 0 |
| Mostly zeros (intermittent demand) | Keep zeros; use intermittent methods (Croston, SBA, TSB, ADIDA, IMAPA) and zero-aware metrics |
| Level shift or structural break | Train on post-break data, add a regime flag, or prefer adaptive models; `ruptures` locates breaks |
| Outlier spikes | Hampel filter or flag as an event; do not smooth away real promotions |
| Short history (under two seasons) | Seasonal models cannot estimate seasonality; prefer global models or foundation models |
| New series with no history | Cold start: use similar series (global models) or a foundation model |

Classify each covariate: **known in advance** (calendar, holidays, planned promotions, prices set
ahead), **observed only up to the origin** (weather, competitor data, the target's own lags), or
**same-period** (inventory at the end of the forecast week: a leak).

For the actual forecast, **generate known-in-advance covariates for the future periods**:
- Calendar-derived ones (week of year, holidays, month ends) are computed from the calendar, never
  filled with 0 or a placeholder. If the history's holiday flag follows a rule (fixed weeks, a
  country calendar via the `holidays` package), reproduce the rule for the horizon.
- Planned inputs nobody has supplied (a promotion calendar) cannot be invented. Forecast a stated
  baseline (for example no promotion), quantify the historical uplift, and offer scenario forecasts.
- Observed-only covariates (weather) need their own forecast or a climatology (per-period averages).

## Step 2: Backtest design

Rolling origins over the training part, each forecasting the full horizon H, leaving the Stage 2
holdout and test periods untouched:

```python
origins = rolling_origins(panel["ds"], horizon=H, n_origins=6, step=4, holdout_last=2 * H)
res = backtest(panel, seasonal_naive(season=52), horizon=H, origins=origins, season=52,
               known_future=["promotion_flag", "public_holiday"])
s = summarize(res)   # overall, by_h, by_series, by_origin
```

- Use at least five origins; the spread of the metric across origins is the noise floor.
- Primary metric: **MASE** (scale-free, comparable across series) or **WAPE** (business-friendly,
  volume-weighted). For probabilistic forecasts: pinball loss, CRPS or weighted quantile loss, plus
  interval coverage. Avoid MAPE with zeros or near-zeros.
- Always report seasonal naive and naive (last value) with the same origins.
- Intermittent series (flagged by `break_report`): WAPE and sMAPE explode near zero even for good
  forecasts. Judge them by MASE and absolute error, report them in their own table, and for
  planning consider a service-level quantity (a reorder point from a high quantile) rather than a
  point forecast.

## Step 3: Candidates

| Family | Methods | Use when |
|---|---|---|
| Baselines | seasonal naive, naive, mean, drift | Always |
| Statistical, local | AutoETS, AutoTheta, AutoARIMA, MSTL (multiple seasonalities), TBATS; `statsforecast` | Few series, clear seasonality, explainability |
| Intermittent | Croston, SBA, TSB, ADIDA, IMAPA (`statsforecast`) | Many zeros |
| ML on lags, global | LightGBM or XGBoost on lags, rolling statistics, calendar and known covariates (`mlforecast`, `skforecast`) | Many related series, covariates, promotions |
| Deep, trained | N-HiTS, N-BEATS, TFT, PatchTST, TiDE, DeepAR (`neuralforecast`, `darts`, `gluonts`) | Very many series, long history, GPU; screen against the benchmark evidence |
| Foundation, zero-shot | Chronos-2, TimesFM-3 or 2.5, TiRex, Moirai-2.0, Toto, TabPFN-TS | Strong default at any number of series; covariate support varies by model |
| Hierarchical | Reconcile base forecasts: bottom-up, top-down, MinTrace (`hierarchicalforecast`) | Totals must equal sums of parts |

**Before a foundation model:** check its weights licence on the model card (code licence is not
the weights licence), whether it runs locally or needs an API (data leaving the machine), GPU needs,
and whether it accepts the covariates the problem depends on. Log all four.

**Direct versus recursive** for lag models: direct (one model per horizon, or lags of at least H)
avoids feeding predictions back in; recursive uses fewer models but compounds errors. With
promotions known in advance, both must see the future promotion flags, never future sales.

**Screening:** run every candidate through the same `backtest` origins, then
`paired_compare({name: result, ...})`: differences are paired by origin, and candidates within one
standard error of the best are kept. Use an origin step of at least the horizon for the final
comparison, so forecast windows do not overlap. Carry the kept candidates, per family, into
`search_space.json`.

## Step 4: Combine and reconcile

Averaging a few diverse forecasters (for example a statistical ensemble plus a foundation model)
is a strong, well-documented strategy; accept it only if the backtest gain exceeds the spread across
origins. If forecasts must sum across a hierarchy (stores to regions to total), reconcile after
forecasting; MinTrace often beats bottom-up when lower levels are noisy. Screen both.

## Step 5: Intervals

Inventory decisions need ranges, not points. Options: model-native quantiles (foundation models,
statistical models, quantile loss for LightGBM), or conformal intervals calibrated on backtest
residuals per horizon. Report empirical coverage per horizon on the backtest; an 80% interval
should contain about 80% of actuals.

## Gate (all must exist)

- gap and break report per series, covariate classification (known / observed / same-period)
- backtest spec: horizon, origins, step, metric; baselines scored on it
- candidate table: metric overall, by horizon, spread across origins; worst series listed
- for foundation models: licence, data-location, hardware and covariate checks, logged
- `search_space.json` updated; decision-log block with skipped families and why

## Red flags

| Thought | Reality |
|---|---|
| "Random K-fold on the weekly rows" | Future rows train the model that predicts the past |
| "Use inventory as a feature" | End-of-week inventory depends on that week's sales: a same-period leak |
| "Average MAPE across stores" | Zeros explode it and small stores dominate; use MASE or WAPE |
| "Deep learning will beat the stats models" | On the public weekly benchmark TFT and PatchTST barely beat seasonal naive |
| "Interpolate all the gaps first" | Interpolation across the origin uses the future; fill within history only |
| "One accuracy number is enough" | Report by horizon and by series; planners need the worst cases |
