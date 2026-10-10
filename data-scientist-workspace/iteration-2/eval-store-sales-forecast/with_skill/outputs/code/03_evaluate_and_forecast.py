"""Stage 8 locked-holdout confirmation, Stage 9 test evaluation (scored once), and the final
production forecast (refit on all available data, 8 weeks beyond the last observed week).
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-store-sales-forecast\with_skill\outputs"
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-forecasting\scripts")
from ts_backtest import seasonal_naive, summarize

sys.path.insert(0, f"{OUT}/code")
from importlib import import_module
mod = import_module("02_backtest_candidates".replace("-", "_")) if False else None

H = 8
SEASON = 52
LAGS = [1, 2, 3, 4, 8, 12]
ROLLS = [4, 8, 12]

panel = pd.read_csv(f"{OUT}/panel_clean.csv", parse_dates=["ds"])
split = pd.read_csv(f"{OUT}/split/split.csv")
panel["row_id"] = panel["unique_id"] + "_" + panel["ds"].dt.strftime("%Y-%m-%d")
panel = panel.merge(split[["id", "part"]], left_on="row_id", right_on="id", how="left").drop(columns=["id", "row_id"])

panel["woy"] = panel["ds"].dt.isocalendar().week.astype(int)
clim_train = panel.loc[panel.part == "train"].groupby(["unique_id", "woy"])["avg_temp_c"].mean().rename("temp_clim")
panel = panel.merge(clim_train, on=["unique_id", "woy"], how="left")
panel["temp_clim"] = panel.groupby("unique_id")["temp_clim"].transform(lambda s: s.fillna(s.mean()))
panel["month"] = panel["ds"].dt.month
panel["woy_sin"] = np.sin(2 * np.pi * panel["woy"] / 52)
panel["woy_cos"] = np.cos(2 * np.pi * panel["woy"] / 52)
KNOWN_FUTURE = ["promo", "holiday_week", "temp_clim", "woy", "month", "woy_sin", "woy_cos"]


def _origin_features(hist_sorted: pd.DataFrame) -> pd.DataFrame:
    g = hist_sorted.copy()
    y = g["y"]
    feats = {f"lag_{L}": y.shift(L) for L in LAGS}
    for R in ROLLS:
        feats[f"rollmean_{R}"] = y.shift(1).rolling(R).mean()
        feats[f"rollstd_{R}"] = y.shift(1).rolling(R).std()
    feats = pd.DataFrame(feats, index=g.index)
    feats["unique_id"] = g["unique_id"].to_numpy()
    feats["ds"] = g["ds"].to_numpy()
    return feats


def fit_lgb(hist: pd.DataFrame, n_estimators=300, num_leaves=15):
    hist = hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rows = []
    for uid, g in hist.groupby("unique_id", sort=False):
        g = g.reset_index(drop=True)
        fe = _origin_features(g)
        for h in range(1, H + 1):
            r = fe.copy()
            r["h"] = h
            r["target"] = g["y"].shift(-h).to_numpy()
            for c in KNOWN_FUTURE:
                r[c] = g[c].shift(-h).to_numpy()
            rows.append(r)
    train = pd.concat(rows, ignore_index=True).dropna(subset=["target"])
    train = train.dropna(subset=[f"lag_{max(LAGS)}"])
    feat_cols = [c for c in train.columns if c not in ("unique_id", "ds", "target")]
    Xtr = train[feat_cols].copy()
    Xtr["unique_id"] = train["unique_id"].astype("category")
    feat_cols2 = feat_cols + ["unique_id"]
    model = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=0.05,
                               min_child_samples=10, verbosity=-1, random_state=0)
    model.fit(Xtr[feat_cols2], train["target"], categorical_feature=["unique_id"])
    return model, feat_cols2, Xtr["unique_id"].cat.categories


def predict_horizon(model, feat_cols2, cats, hist: pd.DataFrame, future_dates_per_store: dict):
    """future_dates_per_store: {unique_id: DataFrame[ds, promo, holiday_week, temp_clim, woy, month, woy_sin, woy_cos]}"""
    hist = hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fe_last = {}
    for uid, g in hist.groupby("unique_id", sort=False):
        g = g.reset_index(drop=True)
        fe_last[uid] = _origin_features(g).iloc[[-1]]
    out_rows = []
    for uid, fut in future_dates_per_store.items():
        base_rows = []
        for h, (_, r) in enumerate(fut.iterrows(), start=1):
            b = fe_last[uid].copy()
            b["h"] = h
            for c in KNOWN_FUTURE:
                b[c] = r[c]
            b["ds_target"] = r["ds"]
            base_rows.append(b)
        Xte = pd.concat(base_rows, ignore_index=True)
        Xte["unique_id"] = uid
        Xte["unique_id"] = Xte["unique_id"].astype("category").cat.set_categories(cats)
        preds = model.predict(Xte[feat_cols2])
        out_rows.append(pd.DataFrame({"unique_id": uid, "ds": Xte["ds_target"].to_numpy(),
                                      "h": range(1, len(fut) + 1), "yhat": preds}))
    return pd.concat(out_rows, ignore_index=True)


def mase_scale(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) <= SEASON:
        return np.nan
    d = np.abs(y[SEASON:] - y[:-SEASON])
    return float(np.mean(d)) if np.mean(d) > 0 else np.nan


def score(y, yhat, scale):
    e = np.abs(np.asarray(y) - np.asarray(yhat))
    mase = np.mean(e / scale)
    wape = e.sum() / np.abs(np.asarray(y)).sum()
    smape = np.mean(2 * e / (np.abs(np.asarray(y)) + np.abs(np.asarray(yhat)) + 1e-9))
    return mase, wape, smape


# ---------------- Stage 8: locked-holdout confirmation ----------------
train = panel[panel.part == "train"].copy()
holdout = panel[panel.part == "holdout"].sort_values(["unique_id", "ds"]).copy()
model, feat_cols2, cats = fit_lgb(train)
fut_map = {uid: g[["ds"] + KNOWN_FUTURE] for uid, g in holdout.groupby("unique_id")}
pred_ho = predict_horizon(model, feat_cols2, cats, train, fut_map)
pred_ho = pred_ho.merge(holdout[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
scale_map = train.groupby("unique_id")["y"].apply(lambda s: mase_scale(s.to_numpy()))
pred_ho["scale"] = pred_ho["unique_id"].map(scale_map)
mase_h, wape_h, smape_h = score(pred_ho["y"], pred_ho["yhat"], pred_ho["scale"])
print(f"[Stage 8 holdout confirm] LightGBM MASE={mase_h:.3f} WAPE={wape_h:.3f} sMAPE={smape_h:.3f}")

sn_ho = holdout.merge(train.groupby("unique_id").tail(SEASON)[["unique_id", "ds", "y"]].rename(columns={"ds": "ds_ref", "y": "y_ref"}), on="unique_id")
# quick seasonal-naive holdout score using the same-week-last-year value
snaive_preds = []
for uid, g in holdout.groupby("unique_id"):
    hist_y = train[train.unique_id == uid].sort_values("ds")["y"].to_numpy()
    n = len(hist_y)
    for j in range(len(g)):
        h = j + 1
        lag = SEASON * int(np.ceil(h / SEASON))
        snaive_preds.append(hist_y[n - lag + h - 1] if n - lag + h - 1 >= 0 else hist_y[-1])
pred_ho["yhat_snaive"] = snaive_preds
mase_sn, wape_sn, smape_sn = score(pred_ho["y"], pred_ho["yhat_snaive"], pred_ho["scale"])
print(f"[Stage 8 holdout confirm] seasonal_naive MASE={mase_sn:.3f} WAPE={wape_sn:.3f} sMAPE={smape_sn:.3f}")

# ---------------- Stage 9: score the frozen test set ONCE ----------------
train_ho = panel[panel.part.isin(["train", "holdout"])].copy()
test = panel[panel.part == "test"].sort_values(["unique_id", "ds"]).copy()
model2, feat_cols2b, cats2 = fit_lgb(train_ho)
fut_map_test = {uid: g[["ds"] + KNOWN_FUTURE] for uid, g in test.groupby("unique_id")}
pred_test = predict_horizon(model2, feat_cols2b, cats2, train_ho, fut_map_test)
pred_test = pred_test.merge(test[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
scale_map2 = train_ho.groupby("unique_id")["y"].apply(lambda s: mase_scale(s.to_numpy()))
pred_test["scale"] = pred_test["unique_id"].map(scale_map2)
pred_test["abs_err"] = (pred_test["y"] - pred_test["yhat"]).abs()
pred_test["ape"] = pred_test["abs_err"] / pred_test["y"].replace(0, np.nan)

mase_t, wape_t, smape_t = score(pred_test["y"], pred_test["yhat"], pred_test["scale"])
print(f"\n[Stage 9 TEST - scored once] overall MASE={mase_t:.3f} WAPE={wape_t:.3f} sMAPE={smape_t:.3f}")

by_h = pred_test.groupby("h").apply(lambda g: pd.Series(dict(zip(["MASE", "WAPE", "sMAPE"], score(g.y, g.yhat, g.scale)))))
print("\nBy horizon:\n", by_h)

by_store = pred_test.groupby("unique_id").apply(lambda g: pd.Series(dict(zip(["MASE", "WAPE", "sMAPE"], score(g.y, g.yhat, g.scale)))))
by_store = by_store.sort_values("WAPE", ascending=False)
print("\nWorst 5 stores by test WAPE:\n", by_store.head(5))
print("\nBest 5 stores by test WAPE:\n", by_store.tail(5))

pred_test.to_csv(f"{OUT}/test_predictions.csv", index=False)
by_h.to_csv(f"{OUT}/test_metrics_by_horizon.csv")
by_store.to_csv(f"{OUT}/test_metrics_by_store.csv")

# per-horizon residual quantiles from the test errors -> simple empirical interval for the forecast
resid = pred_test.copy()
resid["err"] = resid["yhat"] - resid["y"]
q_lo = resid.groupby("h")["err"].quantile(0.1)
q_hi = resid.groupby("h")["err"].quantile(0.9)

# ---------------- Final production forecast: refit on ALL data, 8 weeks beyond last date ----------------
all_data = panel.copy()
model3, feat_cols3, cats3 = fit_lgb(all_data)
last_date = all_data["ds"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=H, freq="W-MON")

# build known-future covariate values for the forecast horizon: promo/holiday assumed 0 (no
# planned-promo calendar supplied - see decision log); temp_clim from the store's own climatology
fut_rows = []
for uid in all_data["unique_id"].unique():
    d = pd.DataFrame({"ds": future_dates})
    d["woy"] = d["ds"].dt.isocalendar().week.astype(int)
    d["month"] = d["ds"].dt.month
    d["woy_sin"] = np.sin(2 * np.pi * d["woy"] / 52)
    d["woy_cos"] = np.cos(2 * np.pi * d["woy"] / 52)
    clim_map = all_data[all_data.unique_id == uid].groupby("woy")["temp_clim"].mean()
    d["temp_clim"] = d["woy"].map(clim_map).fillna(clim_map.mean())
    d["promo"] = 0.0  # ASSUMED: no promo calendar provided for the forecast horizon
    d["holiday_week"] = 0.0  # ASSUMED placeholder; see decision log - recompute from a real calendar before use
    fut_rows.append((uid, d))
fut_map_final = {uid: d for uid, d in fut_rows}
forecast = predict_horizon(model3, feat_cols3, cats3, all_data, fut_map_final)
forecast["q10"] = forecast.apply(lambda r: r["yhat"] + q_lo.get(r["h"], 0), axis=1)
forecast["q90"] = forecast.apply(lambda r: r["yhat"] + q_hi.get(r["h"], 0), axis=1)
forecast = forecast.rename(columns={"unique_id": "store_id", "ds": "week_start", "yhat": "forecast_sales"})
forecast["forecast_sales"] = forecast["forecast_sales"].clip(lower=0)
forecast["q10"] = forecast["q10"].clip(lower=0)
forecast[["store_id", "week_start", "h", "forecast_sales", "q10", "q90"]].to_csv(f"{OUT}/forecast.csv", index=False)
print("\nSaved forecast.csv:", forecast.shape, "date range", future_dates.min(), future_dates.max())
