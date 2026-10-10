"""Stage 8 (ds-method-comparison, compressed) for the forecasting path: screen candidates on
rolling-origin backtests over the train part only (holdout+test periods never touched), pick a
winner, confirm on the locked holdout, then score the frozen test set once.
"""
import sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-store-sales-forecast\with_skill\outputs"
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-forecasting\scripts")
from ts_backtest import rolling_origins, backtest, seasonal_naive, naive_last, summarize

H = 8
SEASON = 52

panel = pd.read_csv(f"{OUT}/panel_clean.csv", parse_dates=["ds"])
split = pd.read_csv(f"{OUT}/split/split.csv")
panel["row_id"] = panel["unique_id"] + "_" + panel["ds"].dt.strftime("%Y-%m-%d")
panel = panel.merge(split[["id", "part"]], left_on="row_id", right_on="id", how="left").drop(columns=["id", "row_id"])
assert panel["part"].isna().sum() == 0

# climatology proxy for future temperature (known in advance, unlike the real future temp):
# week-of-year mean computed on the TRAIN part only, applied to every part.
panel["woy"] = panel["ds"].dt.isocalendar().week.astype(int)
clim = panel.loc[panel.part == "train"].groupby(["unique_id", "woy"])["avg_temp_c"].mean().rename("temp_clim")
panel = panel.merge(clim, on=["unique_id", "woy"], how="left")
panel["temp_clim"] = panel.groupby("unique_id")["temp_clim"].transform(lambda s: s.fillna(s.mean()))
panel["month"] = panel["ds"].dt.month
panel["woy_sin"] = np.sin(2 * np.pi * panel["woy"] / 52)
panel["woy_cos"] = np.cos(2 * np.pi * panel["woy"] / 52)

KNOWN_FUTURE = ["promo", "holiday_week", "temp_clim", "woy", "month", "woy_sin", "woy_cos"]

train_end = panel.loc[panel.part == "train", "ds"].max()
holdout_end = panel.loc[panel.part == "holdout", "ds"].max()
n_holdout_test_periods = panel.loc[panel.part.isin(["holdout", "test"]), "ds"].nunique()
print("train_end", train_end, "holdout_end", holdout_end, "holdout+test periods", n_holdout_test_periods)


# ---------- global LightGBM, direct multi-horizon forecaster ----------
LAGS = [1, 2, 3, 4, 8, 12]
ROLLS = [4, 8, 12]

def _origin_features(hist_sorted: pd.DataFrame) -> pd.DataFrame:
    """Per store, build one feature row per timestamp t using only y[<=t]."""
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


def lgb_direct_forecaster(n_estimators=300, num_leaves=15):
    def f(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        hist = history.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        # training rows: every (store, origin t, h) with a valid target inside history
        rows = []
        for uid, g in hist.groupby("unique_id", sort=False):
            g = g.reset_index(drop=True)
            fe = _origin_features(g)
            for h in range(1, H + 1):
                r = fe.copy()
                r["h"] = h
                r["target"] = g["y"].shift(-h).to_numpy()
                for c in KNOWN_FUTURE:
                    r[c] = g[c].shift(-h).to_numpy()  # covariate value AT the target date (known in advance)
                rows.append(r)
        train = pd.concat(rows, ignore_index=True).dropna(subset=["target"])
        train = train.dropna(subset=[f"lag_{max(LAGS)}"])  # need enough history for the longest lag
        feat_cols = [c for c in train.columns if c not in ("unique_id", "ds", "target")]
        cat_cols = []
        Xtr = train[feat_cols].copy()
        Xtr["unique_id"] = train["unique_id"].astype("category")
        feat_cols2 = feat_cols + ["unique_id"]
        model = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=0.05,
                                   min_child_samples=10, verbosity=-1, random_state=0)
        model.fit(Xtr[feat_cols2], train["target"], categorical_feature=["unique_id"])

        # build features for the actual forecast origin (last row of history per store) and
        # combine with each future row's h + known-future covariates
        preds = np.full(len(future), np.nan)
        fe_last = {}
        for uid, g in hist.groupby("unique_id", sort=False):
            g = g.reset_index(drop=True)
            fe_last[uid] = _origin_features(g).iloc[[-1]]
        fut = future.copy().reset_index(drop=True)
        fut["h"] = fut.groupby("unique_id").cumcount() + 1
        Xte_rows = []
        for i, row in fut.iterrows():
            base = fe_last[row["unique_id"]].copy()
            base["h"] = row["h"]
            for c in KNOWN_FUTURE:
                base[c] = row[c]
            base["unique_id_cat"] = row["unique_id"]
            Xte_rows.append(base)
        Xte = pd.concat(Xte_rows, ignore_index=True)
        Xte["unique_id"] = Xte["unique_id_cat"].astype("category").cat.set_categories(Xtr["unique_id"].cat.categories)
        preds = model.predict(Xte[feat_cols2])
        return np.asarray(preds, dtype=float)
    return f


# ---------- screen on train-only rolling origins (holdout+test untouched) ----------
train_panel = panel[panel.part.isin(["train"])][["unique_id", "ds", "y", "part"] + KNOWN_FUTURE].copy()
# origins must be drawn only from train dates
origins = rolling_origins(train_panel["ds"], horizon=H, n_origins=6, step=4, holdout_last=0)
print("backtest origins:", [str(o.date()) for o in origins])

candidates = {
    "seasonal_naive_52": seasonal_naive(season=SEASON),
    "naive_last": naive_last(),
    "lightgbm_direct_global": lgb_direct_forecaster(),
}

rows_summary = []
by_h_all = []
by_series_all = []
for name, fc in candidates.items():
    res = backtest(train_panel.rename(columns={"y": "y"}), fc, horizon=H, origins=origins, season=SEASON,
                    known_future=KNOWN_FUTURE)
    s = summarize(res)
    overall = s["overall"]
    rows_summary.append({"candidate": name, **overall.to_dict()})
    bh = s["by_h"].reset_index(); bh["candidate"] = name; by_h_all.append(bh)
    bs = s["by_series"].reset_index(); bs["candidate"] = name; by_series_all.append(bs)
    by_origin_mase = s["by_origin"]["MASE"]
    print(name, "overall:", overall.to_dict(), "| MASE spread across origins (sd):", round(by_origin_mase.std(ddof=1), 4))

pd.DataFrame(rows_summary).to_csv(f"{OUT}/candidate_table.csv", index=False)
pd.concat(by_h_all).to_csv(f"{OUT}/candidate_by_horizon.csv", index=False)
pd.concat(by_series_all).to_csv(f"{OUT}/candidate_by_series.csv", index=False)
print("\nSaved candidate_table.csv, candidate_by_horizon.csv, candidate_by_series.csv")
