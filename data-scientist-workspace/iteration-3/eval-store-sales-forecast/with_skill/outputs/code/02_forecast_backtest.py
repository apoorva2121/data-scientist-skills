"""Stages 4-7 (ds-forecasting) + Stage 8 (comparison) for store_sales.

Panel prep, gap/break report, candidate backtest (rolling origins on the TRAIN part only,
holdout/test periods untouched), pick a winner, refit on train+holdout, forecast 8 weeks ahead
for the actual future, and score once on the frozen test set (Stage 9).
"""
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-forecasting\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
from ts_backtest import regularize, rolling_origins, backtest, seasonal_naive, naive_last, summarize, break_report
from cv_folds import load_split, verify_test_untouched

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-store-sales-forecast\with_skill\outputs"
HOLIDAY_WEEKS_ISO = {1, 14, 27, 47, 51, 52}
H = 8
SEASON = 52


def isoweek_holiday_flag(dates: pd.Series) -> pd.Series:
    return dates.dt.isocalendar().week.astype(int).isin(HOLIDAY_WEEKS_ISO).astype(int)

df = pd.read_csv(f"{OUT}/cleaned.csv")
df["week_start"] = pd.to_datetime(df["week_start"])
split = pd.read_csv(f"{OUT}/split/split.csv")
print("test untouched:", verify_test_untouched(f"{OUT}/split"))

# ---- panel prep (Step 1 of ds-forecasting) ----
# regularize() keeps promo/holiday_week/avg_temp_c/inventory_end/row_id from df already; the 3
# "period absent" gap rows get NaN for all of them (no source row) and are fixed below.
panel, gaps = regularize(df, id_col="store_id", time_col="week_start", target="sales", freq="W-MON")
print(f"gaps report: {len(gaps)} rows (period absent vs y-missing):")
print(gaps["gap_type"].value_counts() if len(gaps) else "none")
# 5 nulled sentinel rows (sales==-1) are "row present, y missing" -> interpolate within history
if panel["y"].isna().any():
    panel["y"] = panel.groupby("unique_id")["y"].transform(lambda s: s.interpolate(limit_direction="both"))
    print("interpolated remaining NaNs in y:", panel["y"].isna().sum())
# the 3 period-absent rows: holiday_week is reproduced from the deterministic calendar rule;
# promo has no such rule (ASSUMED no promo on an unrecorded week, the conservative default)
panel["holiday_week"] = isoweek_holiday_flag(panel["ds"])
panel["promo"] = panel["promo"].fillna(0)
panel["row_id"] = panel["unique_id"] + "_" + panel["ds"].dt.strftime("%Y-%m-%d")

breaks = break_report(panel, window=13, ratio=1.3, season=SEASON)
breaks.to_csv(f"{OUT}/break_report.csv", index=False)
print(breaks[breaks.issues != ""])

# attach split part via the row_id used at split time
id2part = split.set_index("id")["part"]
panel["part"] = panel["row_id"].map(id2part).fillna("train")  # regularized filler rows default train (none expected here)

train_panel = panel[panel["part"].isin(["train"])].copy()
holdout_dates = panel.loc[panel["part"] == "holdout", "ds"]
test_dates = panel.loc[panel["part"] == "test", "ds"]
print("train max date:", train_panel["ds"].max(), " holdout:", holdout_dates.min(), "-", holdout_dates.max(),
      " test:", test_dates.min(), "-", test_dates.max())

# ---- Step 2: backtest design (train part only; holdout+test = 16 periods held fully untouched) ----
origins = rolling_origins(train_panel["ds"], horizon=H, n_origins=5, step=4, holdout_last=0)
print("origins:", origins)


def sf_forecaster(model_cls, **kw):
    from statsforecast import StatsForecast

    def f(history, future):
        h = history[["unique_id", "ds", "y"]].dropna()
        sf = StatsForecast(models=[model_cls(season_length=SEASON, **kw)], freq="W-MON", n_jobs=1)
        fc = sf.forecast(df=h, h=H)
        fc = fc.rename(columns={fc.columns[-1]: "yhat"})
        m = future.merge(fc[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="left")
        return m["yhat"].to_numpy()

    return f


def lgbm_global_forecaster():
    """Global, direct multi-horizon LightGBM on lag/rolling features + known covariates.
    Trained fresh inside the forecaster call on `history` only (no future leakage)."""
    import lightgbm as lgb

    def build_rows(hist: pd.DataFrame, h_targets: bool):
        hist = hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        rows = []
        for uid, g in hist.groupby("unique_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            for t in range(SEASON, n):  # need >=52 lookback for lag_52
                lag1, lag52 = g["y"].iloc[t - 1], g["y"].iloc[t - SEASON]
                roll8 = g["y"].iloc[max(0, t - 8):t].mean()
                roll26 = g["y"].iloc[max(0, t - 26):t].mean()
                if h_targets:
                    for h in range(1, H + 1):
                        if t + h - 1 >= n:
                            break
                        fut = g.iloc[t + h - 1]
                        rows.append(dict(unique_id=uid, h=h, lag1=lag1, lag52=lag52, roll8=roll8, roll26=roll26,
                                         promo=fut["promo"], holiday_week=fut["holiday_week"],
                                         woy=fut["ds"].isocalendar()[1], y=fut["y"]))
                else:
                    rows.append(dict(unique_id=uid, t=t, lag1=lag1, lag52=lag52, roll8=roll8, roll26=roll26))
        return pd.DataFrame(rows)

    def f(history, future):
        h = history[["unique_id", "ds", "y", "promo", "holiday_week"]].dropna(subset=["y"])
        train_rows = build_rows(h, h_targets=True)
        feats = ["h", "lag1", "lag52", "roll8", "roll26", "promo", "holiday_week", "woy"]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=30, verbose=-1, random_state=0)
        model.fit(train_rows[feats], train_rows["y"])
        # anchor features at the origin (last row of each series' history)
        anchor = h.sort_values(["unique_id", "ds"]).groupby("unique_id").tail(SEASON)
        last = {}
        for uid, g in anchor.groupby("unique_id"):
            g = g.reset_index(drop=True)
            n = len(g)
            last[uid] = dict(lag1=g["y"].iloc[-1], lag52=g["y"].iloc[0] if n >= SEASON else g["y"].mean(),
                             roll8=g["y"].iloc[-8:].mean(), roll26=g["y"].iloc[-26:].mean())
        fut = future.copy()
        fut["woy"] = fut["ds"].dt.isocalendar().week.astype(int)
        for c in ["lag1", "lag52", "roll8", "roll26"]:
            fut[c] = fut["unique_id"].map(lambda u: last.get(u, {}).get(c, np.nan))
        pred = model.predict(fut[feats])
        return pred

    return f


candidates = {
    "seasonal_naive": seasonal_naive(season=SEASON),
    "naive_last": naive_last(),
    "AutoETS": sf_forecaster(__import__("statsforecast.models", fromlist=["AutoETS"]).AutoETS),
    "AutoTheta": sf_forecaster(__import__("statsforecast.models", fromlist=["AutoTheta"]).AutoTheta),
    "lgbm_global": lgbm_global_forecaster(),
}

results = {}
for name, fc in candidates.items():
    res = backtest(train_panel, fc, horizon=H, origins=origins, season=SEASON,
                   known_future=["promo", "holiday_week"])
    s = summarize(res)
    results[name] = s
    print(f"\n=== {name} === overall:\n{s['overall']}")

overall_tbl = pd.DataFrame({k: v["overall"] for k, v in results.items()}).T
overall_tbl.to_csv(f"{OUT}/candidate_overall.csv")
print("\nCANDIDATE COMPARISON (overall, backtest on train part, 5 origins):")
print(overall_tbl)

by_h = {k: v["by_h"] for k, v in results.items()}
by_origin = {k: v["by_origin"]["MASE"] for k, v in results.items()}
origin_spread = pd.DataFrame(by_origin)
origin_spread.to_csv(f"{OUT}/candidate_by_origin_mase.csv")
print("\nspread across origins (MASE, noise floor):")
print(origin_spread.std())

json.dump({k: {"overall": v["overall"].to_dict()} for k, v in results.items()},
          open(f"{OUT}/candidate_summary.json", "w"), indent=2, default=float)
