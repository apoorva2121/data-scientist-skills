"""
04_backtest.py

Walk-forward backtest of several 8-week-ahead forecasting approaches for
weekly store sales, comparing:
  A) Seasonal naive (lag-52, "same week last year")
  B) Moving-average baseline (mean of last 8 weeks, flat forecast)
  C) Per-store harmonic regression (trend + annual sin/cos + holiday_week)
  D) Global LightGBM, direct multi-horizon (one model per horizon 1..8)
  E) Simple ensemble: average of C and D

Data cleaning applied (see 01_eda.py / 02_eda_followup.py for evidence):
  - 4 exact duplicate (store_id, week_start) rows -> dropped
  - Each store reindexed onto the common 104-week Monday grid
    (2023-01-02 .. 2024-12-23); S03 was missing 3 weeks -> filled by
    time-interpolation of sales, ffill/bfill of temp, holiday_week
    recomputed deterministically from ISO week-of-year, promo set 0
    for the filled weeks (no evidence either way, conservative default).
  - 5 rows with sales == -1 (data-entry sentinel, stores S10/S12) ->
    treated as missing and time-interpolated.
  - Store S19 has a genuine alternating high/zero biweekly sales pattern
    (63/105 weeks are exactly 0, inventory keeps climbing regardless of
    sales) -> kept as-is, it is a real intermittent-demand pattern, not
    an error.

Forecast-time assumptions (see 03_feature_checks.py):
  - holiday_week is 100% deterministic by ISO week-of-year across both
    years in the data (weeks 1, 14, 27, 47, 51, 52) -> safe to compute
    for future weeks from the calendar alone.
  - promo has a strong, consistent positive effect (~15-45% uplift in
    19/20 stores) but no future promo calendar was provided -> future
    promo is assumed 0 ("no promotion planned"). This means the forecast
    is a NO-PROMO baseline; weeks with an actual planned promotion will
    run higher than forecast.
  - avg_temp_c is correlated with sales in most stores but is not known
    8 weeks ahead -> not used directly; its seasonal signal is instead
    captured through calendar (week-of-year) features.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

SRC = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
RNG = np.random.RandomState(0)

HOLIDAY_WOY = {1, 14, 27, 47, 51, 52}
H_MAX = 8

# ---------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------
def load_clean():
    df = pd.read_csv(SRC, parse_dates=["week_start"])
    df = df.drop_duplicates(subset=["store_id", "week_start"]).copy()

    # sentinel negative sales -> missing
    df.loc[df["sales"] < 0, "sales"] = np.nan

    full_grid = pd.date_range("2023-01-02", "2024-12-23", freq="7D")
    assert len(full_grid) == 104

    out = []
    for sid, g in df.groupby("store_id"):
        g = g.set_index("week_start").reindex(full_grid)
        g["store_id"] = sid
        g.index.name = "week_start"
        # interpolate sales (missing weeks + sentinel fix) with time interpolation
        g["sales"] = g["sales"].interpolate(method="linear", limit_direction="both")
        g["avg_temp_c"] = g["avg_temp_c"].interpolate(method="linear", limit_direction="both")
        g["promo"] = g["promo"].fillna(0).astype(int)
        woy = g.index.isocalendar().week.astype(int)
        g["holiday_week"] = woy.isin(HOLIDAY_WOY).astype(int)  # deterministic, recomputed
        g["inventory_end"] = g["inventory_end"].interpolate(method="linear", limit_direction="both")
        out.append(g.reset_index())
    clean = pd.concat(out, ignore_index=True)
    clean["t"] = clean.groupby("store_id").cumcount()  # 0..103 time index per store
    clean["woy"] = clean["week_start"].dt.isocalendar().week.astype(int)
    clean["month"] = clean["week_start"].dt.month
    return clean


# ---------------------------------------------------------------------
# 2. Feature engineering for the ML model
# ---------------------------------------------------------------------
def add_features(df):
    df = df.sort_values(["store_id", "t"]).copy()
    g = df.groupby("store_id")["sales"]
    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_mean_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["roll_mean_8"] = g.shift(1).rolling(8).mean().reset_index(level=0, drop=True)
    df["roll_mean_12"] = g.shift(1).rolling(12).mean().reset_index(level=0, drop=True)
    df["roll_std_4"] = g.shift(1).rolling(4).std().reset_index(level=0, drop=True)
    df["sin1"] = np.sin(2 * np.pi * df["woy"] / 52.0)
    df["cos1"] = np.cos(2 * np.pi * df["woy"] / 52.0)
    df["sin2"] = np.sin(4 * np.pi * df["woy"] / 52.0)
    df["cos2"] = np.cos(4 * np.pi * df["woy"] / 52.0)
    return df


# ---------------------------------------------------------------------
# 3. Candidate forecasters. Each takes (train_df up to & incl. origin t=T,
#    feat_df with features already computed, stores list, T, horizons)
#    and returns a DataFrame [store_id, h, forecast]
# ---------------------------------------------------------------------
def forecast_seasonal_naive(feat_df, T, horizons):
    rows = []
    for sid, g in feat_df.groupby("store_id"):
        g = g.set_index("t")
        for h in horizons:
            target_t = T + h
            src_t = target_t - 52
            if src_t in g.index and src_t <= T:
                val = g.loc[src_t, "sales"]
            else:
                val = g.loc[:T, "sales"].iloc[-1]  # fallback: last observed
            rows.append((sid, h, val))
    return pd.DataFrame(rows, columns=["store_id", "h", "forecast"])


def forecast_moving_avg(feat_df, T, horizons, window=8):
    rows = []
    for sid, g in feat_df.groupby("store_id"):
        hist = g.loc[g["t"] <= T, "sales"]
        val = hist.tail(window).mean()
        for h in horizons:
            rows.append((sid, h, val))
    return pd.DataFrame(rows, columns=["store_id", "h", "forecast"])


def forecast_harmonic(feat_df, T, horizons):
    rows = []
    for sid, g in feat_df.groupby("store_id"):
        hist = g.loc[g["t"] <= T].copy()
        X = hist[["t", "sin1", "cos1", "sin2", "cos2", "holiday_week"]].values
        y = hist["sales"].values
        model = LinearRegression().fit(X, y)
        fut = g.loc[g["t"].isin([T + h for h in horizons])].copy()
        fut = fut.set_index(fut["t"] - T)  # index by h
        Xf = fut[["t", "sin1", "cos1", "sin2", "cos2", "holiday_week"]].values
        preds = model.predict(Xf)
        preds = np.clip(preds, 0, None)
        for h, p in zip(fut.index, preds):
            rows.append((sid, h, p))
    return pd.DataFrame(rows, columns=["store_id", "h", "forecast"])


LGB_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12",
    "roll_mean_4", "roll_mean_8", "roll_mean_12", "roll_std_4",
    "sin1", "cos1", "sin2", "cos2", "holiday_week", "t", "store_id",
]


def forecast_lgbm_direct(feat_df, T, horizons):
    """Direct multi-horizon: one LightGBM model per horizon h, trained
    only on rows whose target time (t+h) <= T (i.e. strictly using data
    available at/if before the origin), predicting for all stores at t=T."""
    df = feat_df.copy()
    df["store_id"] = df["store_id"].astype("category")
    preds_all = []
    for h in horizons:
        train = df[(df["t"] + h <= T) & (df["t"] >= 12)].copy()
        train["target"] = train.groupby("store_id", observed=True)["sales"].shift(-h)
        train = train.dropna(subset=LGB_FEATURES + ["target"])
        if len(train) < 40:
            # not enough history yet for this horizon; fall back to naive
            origin_rows = df[df["t"] == T]
            for _, r in origin_rows.iterrows():
                preds_all.append((r["store_id"], h, r["sales"]))
            continue
        dtrain = lgb.Dataset(train[LGB_FEATURES], label=train["target"],
                              categorical_feature=["store_id"], free_raw_data=False)
        params = dict(
            objective="regression", metric="mae", verbosity=-1,
            num_leaves=15, min_data_in_leaf=15, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            seed=0,
        )
        booster = lgb.train(params, dtrain, num_boost_round=200)
        origin_rows = df[df["t"] == T].copy()
        preds = booster.predict(origin_rows[LGB_FEATURES])
        preds = np.clip(preds, 0, None)
        for sid, p in zip(origin_rows["store_id"], preds):
            preds_all.append((sid, h, p))
    return pd.DataFrame(preds_all, columns=["store_id", "h", "forecast"])


# ---------------------------------------------------------------------
# 4. Backtest loop
# ---------------------------------------------------------------------
def metrics(merged):
    err = merged["forecast"] - merged["actual"]
    wape = err.abs().sum() / merged["actual"].abs().sum()
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    bias = err.mean()
    return pd.Series({"WAPE": wape, "MAE": mae, "RMSE": rmse, "BIAS": bias, "n": len(merged)})


def main():
    clean = load_clean()
    feat = add_features(clean)

    horizons = list(range(1, H_MAX + 1))
    origins = [60, 68, 76, 84, 92]  # walk-forward origins (t index), leave room for +8 <= 103

    models = {
        "A_seasonal_naive": forecast_seasonal_naive,
        "B_moving_avg_8w": forecast_moving_avg,
        "C_harmonic_reg": forecast_harmonic,
        "D_lgbm_direct": forecast_lgbm_direct,
    }

    all_results = {}
    per_horizon = {}
    per_store = {}

    actual_lookup = feat.set_index(["store_id", "t"])["sales"]

    for name, fn in models.items():
        preds_by_origin = []
        for T in origins:
            p = fn(feat, T, horizons)
            p["T"] = T
            preds_by_origin.append(p)
        preds = pd.concat(preds_by_origin, ignore_index=True)
        preds["target_t"] = preds["T"] + preds["h"]
        preds["actual"] = preds.apply(lambda r: actual_lookup.get((r["store_id"], r["target_t"]), np.nan), axis=1)
        preds = preds.dropna(subset=["actual"])
        all_results[name] = metrics(preds)
        per_horizon[name] = preds.groupby("h").apply(metrics)
        per_store[name] = preds.groupby("store_id").apply(metrics)
        preds.to_csv(rf"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\_debug_preds_{name}.csv", index=False)

    # ensemble E = average of C and D
    predsC = pd.concat([forecast_harmonic(feat, T, horizons).assign(T=T) for T in origins], ignore_index=True)
    predsD = pd.concat([forecast_lgbm_direct(feat, T, horizons).assign(T=T) for T in origins], ignore_index=True)
    m = predsC.merge(predsD, on=["store_id", "h", "T"], suffixes=("_C", "_D"))
    m["forecast"] = 0.5 * m["forecast_C"] + 0.5 * m["forecast_D"]
    m["target_t"] = m["T"] + m["h"]
    m["actual"] = m.apply(lambda r: actual_lookup.get((r["store_id"], r["target_t"]), np.nan), axis=1)
    m = m.dropna(subset=["actual"])
    all_results["E_ensemble_CD"] = metrics(m)
    per_horizon["E_ensemble_CD"] = m.groupby("h").apply(metrics)
    per_store["E_ensemble_CD"] = m.groupby("store_id").apply(metrics)

    print("=== Overall backtest metrics (5 origins x 8 horizons x 20 stores) ===")
    summary = pd.DataFrame(all_results).T
    print(summary.to_string())

    print("\n=== WAPE by horizon (1..8 weeks ahead) ===")
    wape_by_h = pd.DataFrame({name: per_horizon[name]["WAPE"] for name in models})
    wape_by_h["E_ensemble_CD"] = per_horizon["E_ensemble_CD"]["WAPE"]
    print(wape_by_h.to_string())

    print("\n=== Best model (D_lgbm_direct) WAPE by store ===")
    print(per_store["D_lgbm_direct"]["WAPE"].sort_values(ascending=False).to_string())

    print("\n=== Harmonic reg (C) WAPE by store ===")
    print(per_store["C_harmonic_reg"]["WAPE"].sort_values(ascending=False).to_string())

    print("\n=== Ensemble (E) WAPE by store ===")
    print(per_store["E_ensemble_CD"]["WAPE"].sort_values(ascending=False).to_string())

    summary.to_csv(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\_backtest_summary.csv")
    wape_by_h.to_csv(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\_backtest_wape_by_horizon.csv")
    per_store["D_lgbm_direct"].to_csv(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\_backtest_per_store_lgbm.csv")
    per_store["E_ensemble_CD"].to_csv(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-store-sales-forecast\without_skill\outputs\_backtest_per_store_ensemble.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
