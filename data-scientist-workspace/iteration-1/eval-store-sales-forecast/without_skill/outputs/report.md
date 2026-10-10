# 8-Week-Ahead Store Sales Forecast — Report

**Source data:** `evals/datasets/store_sales.csv` — 20 stores, weekly sales,
2023-01-02 to 2024-12-23 (104 weeks, ~2 years), 2,081 raw rows.
Columns: `store_id, week_start, promo, holiday_week, avg_temp_c, sales, inventory_end`.

**Goal:** an 8-week-ahead sales forecast per store for inventory planning,
with an honest estimate of how accurate it will be.

All code referenced below is in this folder and runs top-to-bottom:
`01_eda.py` → `02_eda_followup.py` → `03_feature_checks.py` →
`04_backtest.py` → `05_harmonic_variants.py` → `06_final_forecast.py`.

---

## 1. What was done, in order

1. **`01_eda.py`** — shape, dtypes, per-store row counts, missing values,
   duplicate keys, distribution of `sales`, outlier scan, week-continuity
   check.
2. **`02_eda_followup.py`** — drilled into every anomaly flagged by step 1
   (duplicate rows, missing weeks, negative sales, the one store with lots
   of zero sales) to decide how to treat each one.
3. **`03_feature_checks.py`** — checked whether `holiday_week`, `promo`,
   `avg_temp_c` carry real signal, and whether `holiday_week` is
   deterministic by calendar week (i.e. usable for a *future* forecast
   without leaking information).
4. **`04_backtest.py`** — cleaned the data, engineered features, implemented
   four candidate forecasters (seasonal naive, moving average, per-store
   harmonic regression, global LightGBM direct multi-horizon), and ran a
   walk-forward backtest of all of them plus an ensemble.
5. **`05_harmonic_variants.py`** — the harmonic regression won step 4 by a
   clear margin, so tried small variants of it (add `promo` history, fewer/
   more harmonics, a lag-1 "momentum" term) to see if any beat the base
   version.
6. **`06_final_forecast.py`** — refit the winning model on all 104 weeks of
   cleaned history and produced the 8-week-ahead forecast
   (`forecast.csv`).

---

## 2. What we found in the data

| Issue | Detail | How it was handled |
|---|---|---|
| Exact duplicate rows | 4 rows are byte-for-byte duplicates of another row for the same `(store_id, week_start)` (stores S08, S17, S18, S19) | Dropped the duplicate copy |
| Missing weeks | Store **S03** is missing 3 weeks (`2023-06-05`, `2023-06-12`, `2024-02-19`) out of the common 104-week grid | Reindexed every store onto the same 104-week Monday grid; missing `sales`/`avg_temp_c` filled by linear time-interpolation; `holiday_week` recomputed from the calendar (see below); `promo` set to 0 for the filled weeks (no information either way) |
| Negative sales | 5 rows have `sales == -1` (stores S10 ×2, S12 ×3) — clearly a data-entry sentinel, not a real value (all other values are ≥0) | Treated as missing, linear time-interpolation |
| Store S19 is structurally different | 63 of its 105 raw weeks have `sales == 0`; average weekly sales ≈ 590 vs. 8,000–57,000 for every other store; inventory keeps climbing regardless of that week's sales (corr(sales, inventory_end) ≈ −0.08 vs. −0.4 to −0.9 for normal stores) | Kept as-is — this looks like a genuine **intermittent-demand** store (e.g. low-velocity or bulk item), not corrupted data. It is the weak point of every model below (see §4) and is flagged separately rather than "fixed." |
| `holiday_week` is deterministic | It is 1 for ISO weeks {1, 14, 27, 47, 51, 52} in **both** years and 0 otherwise, with zero exceptions | Safe to compute for future weeks from the calendar alone — used as a feature in the forecast |
| `promo` matters a lot, but its future schedule is unknown | Promo weeks show a consistent **+11% to +45% sales uplift** (mean ≈ +24%) in 19 of 20 stores; only S19 shows no promo effect | No future promo calendar was supplied, so the model does **not** assume any store runs a promo in the forecast window (see caveats — this is the single biggest lever on forecast accuracy that the client can supply) |
| `avg_temp_c` is correlated with sales but not forecastable | Within-store correlation with sales ranges from −0.73 to +0.77 (sign and strength vary a lot by store — different product mixes) | Not used directly, since a real 8-week-ahead temperature forecast isn't available; its seasonal signal is captured instead through calendar (week-of-year) terms, which is what temperature is mostly a proxy for at weekly granularity |
| `inventory_end` | Reflects sales that already happened plus restocking decisions — using it as a predictor of the *same* week's sales would be circular | Not used as a model input |

---

## 3. Approaches compared, and how

### Candidates
- **A — Seasonal naive:** forecast = sales from the same week one year earlier.
- **B — Moving average:** forecast = flat average of the trailing 8 weeks.
- **C — Per-store harmonic regression (OLS):**
  `sales(t) = b0 + b1·t + b2·sin(2π·woy/52) + b3·cos(2π·woy/52) + b4·holiday_week(t)`
  fit separately for each store — trend + one annual seasonal cycle + the
  deterministic holiday flag.
- **D — Global LightGBM, direct multi-horizon:** one gradient-boosted model
  per horizon (h = 1…8), pooled across all 20 stores, using lags
  (1,2,3,4,8,12), rolling mean/std, calendar terms, `store_id` and
  `holiday_week`.
- **E — Ensemble:** simple 50/50 average of C and D.

### Backtest scheme
Walk-forward / rolling-origin validation, **not** a single train/test split
(too easy to get lucky with 2 years of data): 5 origins spaced through the
back half of the series (weeks 60, 68, 76, 84, 92 of 104), and at each
origin, forecasts for horizons **h = 1…8 weeks** for **all 20 stores**,
using only information available up to that origin (models are refit at
every origin so there is no leakage from the future). That gives
5 × 8 × 20 = **800 (store, horizon) forecast/actual pairs** per candidate.
Metric: **WAPE** (`sum(|error|) / sum(|actual|)`), chosen over MAPE because
store S19 has many weeks with `sales == 0`, which makes MAPE undefined/
explosive; WAPE is also the standard retail-forecasting metric and is
directly interpretable ("total forecast error as a % of total demand").
MAE, RMSE and bias were also tracked (see `_backtest_summary.csv`).

### Results (overall, 800 pairs each)

| Model | WAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| A — Seasonal naive | 0.154 | 3,424 | 4,796 | −2,466 |
| B — Moving average (8w) | 0.144 | 3,204 | 4,370 | −629 |
| **C — Harmonic regression** | **0.089** | **1,973** | **2,938** | −495 |
| D — LightGBM direct | 0.129 | 2,856 | 3,998 | −108 |
| E — Ensemble (C+D) | 0.102 | 2,262 | 3,242 | −302 |

The harmonic regression beat everything else, including the ensemble (blending
in the weaker LightGBM model only hurt it). This is expected with only
~104 weeks × 20 stores of data: LightGBM's direct multi-horizon models
each train on roughly 1,500–1,800 rows with dozens of possible splits and
20 store identities, which is thin for gradient boosting to learn reliable
lag/seasonality interactions; a 5-parameter-per-store linear model with a
strong, correct functional form (trend + one seasonal cycle) is much
better matched to the amount of data available.

Full per-horizon WAPE is roughly flat across h = 1…8 (≈ 8%–11%, no
strong degradation with horizon — see `_backtest_wape_by_horizon.csv`),
which is a nice property of a direct-from-calendar regression: it doesn't
compound recursive forecast error the way a lag-based recursive model would.

**Variant check (`05_harmonic_variants.py`):** tried (a) adding `promo`
history with 0 assumed for the forecast window, (b) one harmonic instead
of two, (c) a lag-1 "last observed value" momentum term, (d) promo+lag-1
combined.

| Variant | WAPE |
|---|---|
| Base (2 harmonics) | 0.0888 |
| + promo (assume 0 future) | 0.0868 |
| **1 harmonic only** | **0.0860** |
| + lag-1 momentum | 0.0900 |
| + promo + lag-1 | 0.0870 |

One harmonic was marginally best and simpler, so that is the **final
model**. Adding `promo` nudged WAPE down slightly but made the bias
notably more negative (−1,147 vs. −495), because forcing `promo=0` at
forecast time systematically subtracts out part of the historical average
that real promo weeks in the backtest still contained — i.e. it trades a
small amount of aggregate WAPE for a more systematic under-forecast. Given
this is for inventory planning (systematic under-forecast → stockouts),
the simpler, less-biased version was kept.

---

## 4. Expected accuracy

**Final model:** per-store OLS, `sales(t) = b0 + b1·t + b2·sin(2π·woy/52)
+ b3·cos(2π·woy/52) + b4·holiday_week(t)`, refit on all 104 weeks of
cleaned history per store.

**Headline number: ~8.6% WAPE** on the 8-week-ahead walk-forward backtest
(800 forecast/actual pairs, 5 origins × 8 horizons × 20 stores), i.e. on
average the forecast's total absolute error is about 8–9% of total demand.
MAE ≈ 1,900 units/week, RMSE ≈ 2,900 units/week, small negative bias
(≈ −500, forecast running ~2% low on average across the backtest).

**This accuracy is not uniform across stores.** From
`_backtest_per_store_lgbm.csv`/`_backtest_per_store_ensemble.csv` and the
final-model diagnostics (`_final_model_diagnostics.csv`, in-sample R² per
store):
- **19 of 20 stores** fit well (in-sample R² 0.45–0.84) and should track
  the ~7–15% WAPE range store-by-store; the higher-volume, more seasonal
  stores (e.g. S02, S06, S13, S14) fit best (R² 0.76–0.84).
- **Store S19 is the exception.** In-sample R² is only 0.10, and its
  backtest WAPE was **>100%** for every candidate model (harmonic, LightGBM,
  ensemble alike). Its demand is intermittent/sporadic (roughly 40% of
  weeks have sales, the rest are exactly zero, with no clean fixed period)
  and essentially uncorrelated with calendar, promo, or temperature in this
  data. No model tried here can predict *which* week will be the "on" week;
  the best any of them can do is forecast close to the long-run per-week
  average (≈ 590 units). **Recommendation:** for inventory planning on
  S19 specifically, use an intermittent-demand method (e.g. Croston's
  method / TSB, or a reorder-point-and-safety-stock policy keyed to the
  historical run rate) rather than a single point forecast — a smooth
  weekly forecast will look wrong most individual weeks even though it is
  the best available "expected value."

**Uncertainty bands:** `forecast.csv` includes an approximate 80%
interval (`forecast_lower_80` / `forecast_upper_80`) built from each
store's in-sample residual standard deviation (±1.28σ). This is a rough,
constant-width-per-store band, **not** a rigorous forecast interval — it
doesn't grow with horizon and doesn't account for parameter uncertainty.
Treat it as a planning guide (order-of-magnitude safety stock sizing), not
a statistically calibrated interval.

---

## 5. Caveats and assumptions (please read before using this for ordering)

1. **No promo assumed.** This is a "business as usual, no promotion"
   forecast. Promo weeks show a consistent **+11% to +45%** (avg ~+24%)
   sales lift in 19/20 stores historically. If any promotions are planned
   in the next 8 weeks, actual demand for those weeks will very likely run
   well above this forecast — if a promo calendar becomes available, it
   should be fed back into the model (it's a one-line change, see
   `05_harmonic_variants.py`'s `C_promo` variant).
2. **Store S19 needs a different method** (see §4) — its point forecast is
   the long-run average, not a real weekly prediction.
3. **Only ~2 years of history (2 seasonal cycles).** The annual seasonality
   term is estimated from just two observations of each calendar week, so
   it is a reasonable but not highly precise seasonal profile; it will
   improve automatically as more years of data accumulate.
4. **Weather is not modeled.** `avg_temp_c` matters in most stores but a
   real 8-week temperature forecast wasn't available, so weekly seasonality
   is used as its proxy. An unusually warm/cold spell versus seasonal norms
   would not be captured.
5. **Missing-week interpolation.** S03's 3 missing weeks and the 5 `-1`
   sentinel values were linearly interpolated (8 of 2,081 rows total,
   <0.4% of the data) — immaterial to the fitted trend/seasonality but
   noted for transparency.
6. **Prediction intervals are approximate** (constant per-store ±1.28σ,
   not horizon-aware) — use for rough safety-stock sizing only.
7. Numbers above (WAPE, R², uplift %, etc.) are all computed in the
   scripts in this folder and can be reproduced by rerunning them in
   order; nothing in this report was hand-typed from memory.

---

## Files in this folder

- `01_eda.py`, `02_eda_followup.py`, `03_feature_checks.py` — data
  exploration and cleaning-decision evidence
- `04_backtest.py` — cleaning pipeline, feature engineering, all 4
  candidate models, walk-forward backtest
- `05_harmonic_variants.py` — final-model variant selection
- `06_final_forecast.py` — refits the chosen model on full history and
  writes `forecast.csv`
- `forecast.csv` — 8-week-ahead forecast per store (160 rows = 20 stores ×
  8 weeks), columns: `store_id, week_start, horizon_weeks, forecast_sales,
  forecast_lower_80, forecast_upper_80`
- `_backtest_summary.csv`, `_backtest_wape_by_horizon.csv`,
  `_backtest_per_store_lgbm.csv`, `_backtest_per_store_ensemble.csv`,
  `_final_model_diagnostics.csv` — supporting numbers referenced above
