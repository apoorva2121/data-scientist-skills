"""Stage 8 (holdout check) + Stage 9 (test scoring, once) + Stage 10 (future forecast) for
store_sales. Picked model: lgbm_global (see decision_log.md Stage 8) -- tied with AutoTheta and
naive_last on backtest MASE (paired diffs << paired SE across 5 origins), but it is the only
candidate that uses the known-in-advance covariates promo/holiday_week, and the actual 8-week
target window (2024-11-04..2024-12-23) contains 3 holiday weeks with a measured +18% median
uplift, and promo has a measured +23% median uplift -- covariate-awareness matters specifically
for this horizon, unlike most of the backtest origins.
"""
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-forecasting\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
from ts_backtest import regularize, seasonal_naive, naive_last
from cv_folds import verify_test_untouched

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-store-sales-forecast\with_skill\outputs"
HOLIDAY_WEEKS_ISO = {1, 14, 27, 47, 51, 52}
H = 8
SEASON = 52


def isoweek_holiday_flag(dates: pd.Series) -> pd.Series:
    return dates.dt.isocalendar().week.astype(int).isin(HOLIDAY_WEEKS_ISO).astype(int)


def build_rows(hist: pd.DataFrame):
    hist = hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rows = []
    for uid, g in hist.groupby("unique_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        for t in range(SEASON, n):
            lag1, lag52 = g["y"].iloc[t - 1], g["y"].iloc[t - SEASON]
            roll8 = g["y"].iloc[max(0, t - 8):t].mean()
            roll26 = g["y"].iloc[max(0, t - 26):t].mean()
            for h in range(1, H + 1):
                if t + h - 1 >= n:
                    break
                fut = g.iloc[t + h - 1]
                rows.append(dict(unique_id=uid, h=h, lag1=lag1, lag52=lag52, roll8=roll8, roll26=roll26,
                                 promo=fut["promo"], holiday_week=fut["holiday_week"],
                                 woy=fut["ds"].isocalendar()[1], y=fut["y"]))
    return pd.DataFrame(rows)


FEATS = ["h", "lag1", "lag52", "roll8", "roll26", "promo", "holiday_week", "woy"]


def fit_lgbm(hist: pd.DataFrame):
    train_rows = build_rows(hist)
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=30, verbose=-1, random_state=0)
    model.fit(train_rows[FEATS], train_rows["y"])
    return model


def anchor_features(hist: pd.DataFrame):
    anchor = hist.sort_values(["unique_id", "ds"]).groupby("unique_id").tail(SEASON)
    last = {}
    for uid, g in anchor.groupby("unique_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        last[uid] = dict(lag1=g["y"].iloc[-1], lag52=g["y"].iloc[0] if n >= SEASON else g["y"].mean(),
                         roll8=g["y"].iloc[-8:].mean(), roll26=g["y"].iloc[-26:].mean())
    return last


def predict_h(model, hist, future_dates, promo_assumption=0):
    last = anchor_features(hist)
    rows = []
    for uid in hist["unique_id"].unique():
        for h, ds in enumerate(future_dates, start=1):
            f = last.get(uid, {})
            rows.append(dict(unique_id=uid, ds=ds, h=h, lag1=f.get("lag1", np.nan), lag52=f.get("lag52", np.nan),
                             roll8=f.get("roll8", np.nan), roll26=f.get("roll26", np.nan),
                             promo=promo_assumption, holiday_week=isoweek_holiday_flag(pd.Series([ds])).iloc[0],
                             woy=ds.isocalendar()[1]))
    fut = pd.DataFrame(rows)
    fut["yhat"] = model.predict(fut[FEATS])
    return fut


# ---- reload panel (same construction as script 02) ----
df = pd.read_csv(f"{OUT}/cleaned.csv")
df["week_start"] = pd.to_datetime(df["week_start"])
split = pd.read_csv(f"{OUT}/split/split.csv")
assert verify_test_untouched(f"{OUT}/split")

panel, gaps = regularize(df, id_col="store_id", time_col="week_start", target="sales", freq="W-MON")
panel["y"] = panel.groupby("unique_id")["y"].transform(lambda s: s.interpolate(limit_direction="both"))
panel["holiday_week"] = isoweek_holiday_flag(panel["ds"])
panel["promo"] = panel["promo"].fillna(0)
panel["row_id"] = panel["unique_id"] + "_" + panel["ds"].dt.strftime("%Y-%m-%d")
id2part = split.set_index("id")["part"]
panel["part"] = panel["row_id"].map(id2part).fillna("train")

train_holdout = panel[panel["part"].isin(["train", "holdout"])].copy()   # everything before test
full_hist = panel.copy()                                                  # everything (for the real future forecast)
test_rows = panel[panel["part"] == "test"].copy()

# ---- Stage 8: does the pick beat/hold vs holdout, refit on train only, predict holdout ----
model_holdout_check = fit_lgbm(panel[panel["part"] == "train"])
holdout_dates = sorted(panel.loc[panel["part"] == "holdout", "ds"].unique())
pred_ho = predict_h(model_holdout_check, panel[panel["part"] == "train"], pd.to_datetime(holdout_dates))
ho_eval = panel[panel["part"] == "holdout"][["unique_id", "ds", "y"]].merge(pred_ho[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"])
ho_wape = (ho_eval["y"] - ho_eval["yhat"]).abs().sum() / ho_eval["y"].abs().sum()
print("Stage 8 holdout WAPE (lgbm_global, refit on train only):", round(ho_wape, 4))

# ---- Stage 9: final refit on train+holdout, score test ONCE ----
final_model = fit_lgbm(train_holdout)
test_dates = sorted(test_rows["ds"].unique())
pred_test = predict_h(final_model, train_holdout, pd.to_datetime(test_dates))
test_eval = test_rows[["unique_id", "ds", "y", "promo", "holiday_week"]].merge(
    pred_test[["unique_id", "ds", "h", "yhat"]], on=["unique_id", "ds"])

# scale for MASE from each series' TRAIN+HOLDOUT history (never test)
scale = train_holdout.groupby("unique_id")["y"].apply(
    lambda s: np.mean(np.abs(s.to_numpy()[SEASON:] - s.to_numpy()[:-SEASON])))
test_eval["scale"] = test_eval["unique_id"].map(scale)
test_eval["ae"] = (test_eval["y"] - test_eval["yhat"]).abs()

# baselines on the same test rows, same origin (train+holdout as history)
sn = seasonal_naive(season=SEASON)
nl = naive_last()
fut_cols = train_holdout[["unique_id", "ds"]].copy()
future_frame = test_eval[["unique_id", "ds", "h"]].copy()
yhat_sn = sn(train_holdout[["unique_id", "ds", "y"]], future_frame)
yhat_nl = nl(train_holdout[["unique_id", "ds", "y"]], future_frame)
test_eval["yhat_seasonal_naive"] = yhat_sn
test_eval["yhat_naive_last"] = yhat_nl
test_eval["ae_sn"] = (test_eval["y"] - test_eval["yhat_seasonal_naive"]).abs()
test_eval["ae_nl"] = (test_eval["y"] - test_eval["yhat_naive_last"]).abs()

def report_metrics(ae_col):
    ae = test_eval[ae_col]
    mase = (ae / test_eval["scale"]).mean()
    wape = ae.sum() / test_eval["y"].abs().sum()
    smape = (2 * ae / (test_eval["y"].abs() + test_eval[ae_col.replace("ae", "yhat") if "yhat" in ae_col else "yhat"].abs())).mean() if False else np.nan
    return mase, wape

for name, col in [("lgbm_global (final pick)", "ae"), ("seasonal_naive", "ae_sn"), ("naive_last", "ae_nl")]:
    mase, wape = report_metrics(col)
    print(f"TEST {name}: MASE={mase:.4f} WAPE={wape:.4f}")

# bootstrap CI (by series, 2000 resamples) for the pick's WAPE and MASE
rng = np.random.default_rng(42)
series_ids = test_eval["unique_id"].unique()
boot_mase, boot_wape = [], []
for _ in range(2000):
    samp = rng.choice(series_ids, size=len(series_ids), replace=True)
    d = pd.concat([test_eval[test_eval.unique_id == s] for s in samp])
    boot_mase.append((d["ae"] / d["scale"]).mean())
    boot_wape.append(d["ae"].sum() / d["y"].abs().sum())
mase_ci = (np.percentile(boot_mase, 2.5), np.percentile(boot_mase, 97.5))
wape_ci = (np.percentile(boot_wape, 2.5), np.percentile(boot_wape, 97.5))
print("MASE 95% CI (block bootstrap by store):", mase_ci)
print("WAPE 95% CI (block bootstrap by store):", wape_ci)

by_h_test = test_eval.groupby("h").apply(lambda g: pd.Series({
    "MASE": (g["ae"] / g["scale"]).mean(), "WAPE": g["ae"].sum() / g["y"].abs().sum(), "n": len(g)}))
print("\nTEST by horizon (lgbm_global):\n", by_h_test)

by_store_test = test_eval.groupby("unique_id").apply(lambda g: pd.Series({
    "MASE": (g["ae"] / g["scale"]).mean(), "WAPE": g["ae"].sum() / g["y"].abs().sum(), "n": len(g)})).sort_values("MASE", ascending=False)
print("\nTEST by store (lgbm_global), worst first:\n", by_store_test.head(6))
print("...best:\n", by_store_test.tail(3))

by_h_test.to_csv(f"{OUT}/test_by_horizon.csv")
by_store_test.to_csv(f"{OUT}/test_by_store.csv")
test_eval.to_csv(f"{OUT}/test_predictions.csv", index=False)

# ---- Stage 10: true future forecast, refit on ALL data (train+holdout+test) ----
model_prod = fit_lgbm(full_hist)
last_date = full_hist["ds"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
fc = predict_h(model_prod, full_hist, future_dates, promo_assumption=0)
fc = fc.rename(columns={"unique_id": "store_id"})[["store_id", "ds", "h", "yhat", "holiday_week", "promo"]]
fc = fc.rename(columns={"ds": "week_start", "yhat": "sales_forecast"})
fc["sales_forecast"] = fc["sales_forecast"].round(0)
fc.to_csv(f"{OUT}/forecast.csv", index=False)
print("\nforecast.csv written:", fc.shape)
print(fc.head(10))

json.dump({
    "holdout_wape": float(ho_wape),
    "test_mase_pick": float(report_metrics("ae")[0]), "test_wape_pick": float(report_metrics("ae")[1]),
    "test_mase_seasonal_naive": float(report_metrics("ae_sn")[0]), "test_wape_seasonal_naive": float(report_metrics("ae_sn")[1]),
    "test_mase_naive_last": float(report_metrics("ae_nl")[0]), "test_wape_naive_last": float(report_metrics("ae_nl")[1]),
    "test_mase_ci95": list(mase_ci), "test_wape_ci95": list(wape_ci),
}, open(f"{OUT}/final_test_summary.json", "w"), indent=2, default=float)
