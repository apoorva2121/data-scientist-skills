"""
06_final_forecast.py

Final model, chosen from the walk-forward backtest in 04_backtest.py /
05_harmonic_variants.py:

  Per-store OLS "harmonic regression":
    sales(t) = b0 + b1*t + b2*sin(2*pi*woy/52) + b3*cos(2*pi*woy/52)
               + b4*holiday_week(t) + eps

  This beat seasonal-naive, an 8-week moving average, and a global
  LightGBM direct multi-horizon model in backtest WAPE (0.086 vs
  0.144-0.154 for the naive baselines and 0.129 for LightGBM), and did
  so with far fewer parameters (5 per store) which matters given only
  ~2 years / 104 weeks of history per store. Adding promo history or a
  lag-1 "momentum" term did not reliably improve on this in backtest,
  so the final model is kept at this simple form (see 05 for the
  comparison numbers).

Refit on ALL cleaned history (104 weeks/store) and used to forecast the
next 8 weeks (t = 104..111, i.e. 2024-12-30 .. 2025-02-17) for every
store. Prediction intervals are an approximate 80% band using each
store's in-sample residual std (not a rigorous forecast-uncertainty
model -- see report.md caveats).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import importlib.util
spec = importlib.util.spec_from_file_location("bt", r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\04_backtest.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

OUT_DIR = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs"
HOLIDAY_WOY = {1, 14, 27, 47, 51, 52}

clean = bt.load_clean()
feat = bt.add_features(clean)

last_t = feat["t"].max()  # 103
last_date = feat["week_start"].max()
future_t = list(range(last_t + 1, last_t + 9))  # 104..111
future_dates = [last_date + pd.Timedelta(days=7 * h) for h in range(1, 9)]

rows = []
diag_rows = []
for sid, g in feat.groupby("store_id"):
    g = g.sort_values("t")
    X = g[["t", "sin1", "cos1", "holiday_week"]].values
    y = g["sales"].values
    model = LinearRegression().fit(X, y)
    resid = y - model.predict(X)
    resid_std = resid.std(ddof=5)  # ddof ~ n_params
    r2 = model.score(X, y)

    fut = pd.DataFrame({"t": future_t, "week_start": future_dates})
    woy = fut["week_start"].dt.isocalendar().week.astype(int)
    fut["sin1"] = np.sin(2 * np.pi * woy / 52.0)
    fut["cos1"] = np.cos(2 * np.pi * woy / 52.0)
    fut["holiday_week"] = woy.isin(HOLIDAY_WOY).astype(int)
    Xf = fut[["t", "sin1", "cos1", "holiday_week"]].values
    preds = np.clip(model.predict(Xf), 0, None)

    for h, (t, wk, p) in enumerate(zip(future_t, future_dates, preds), start=1):
        lower = max(0.0, p - 1.2816 * resid_std)
        upper = p + 1.2816 * resid_std
        rows.append({
            "store_id": sid,
            "week_start": wk.date().isoformat(),
            "horizon_weeks": h,
            "forecast_sales": round(float(p), 1),
            "forecast_lower_80": round(float(lower), 1),
            "forecast_upper_80": round(float(upper), 1),
        })
    diag_rows.append({"store_id": sid, "r2_in_sample": round(r2, 3),
                       "resid_std": round(resid_std, 1),
                       "mean_sales": round(g["sales"].mean(), 1),
                       "n_obs": len(g)})

forecast_df = pd.DataFrame(rows)
forecast_df.to_csv(rf"{OUT_DIR}\forecast.csv", index=False)

diag_df = pd.DataFrame(diag_rows).sort_values("r2_in_sample")
diag_df.to_csv(rf"{OUT_DIR}\_final_model_diagnostics.csv", index=False)

print("Forecast horizon dates:", [d.date().isoformat() for d in future_dates])
print("\nforecast.csv written, shape:", forecast_df.shape)
print(forecast_df.head(16).to_string())

print("\n=== per-store in-sample fit diagnostics ===")
print(diag_df.to_string())

print("\n=== sanity: total forecast sales across all stores per horizon week ===")
print(forecast_df.groupby("week_start")["forecast_sales"].sum())

print("\nDone.")
