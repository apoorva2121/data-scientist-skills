"""
05_harmonic_variants.py
Quick variants on the winning harmonic regression (per-store trend + annual
sin/cos + holiday_week) from 04_backtest.py, to see if adding promo history
(with promo=0 assumed at forecast time), an extra harmonic, or a lag-1
"momentum" term improves the walk-forward backtest.
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

clean = bt.load_clean()
feat = bt.add_features(clean)
horizons = list(range(1, 9))
origins = [60, 68, 76, 84, 92]
actual_lookup = feat.set_index(["store_id", "t"])["sales"]


def metrics(merged):
    err = merged["forecast"] - merged["actual"]
    wape = err.abs().sum() / merged["actual"].abs().sum()
    return pd.Series({"WAPE": wape, "MAE": err.abs().mean(), "RMSE": np.sqrt((err**2).mean()), "BIAS": err.mean(), "n": len(merged)})


def forecast_harmonic_v2(feat_df, T, horizons, use_promo=False, n_harm=2, use_lag1=False):
    rows = []
    for sid, g in feat_df.groupby("store_id"):
        hist = g.loc[g["t"] <= T].copy()
        cols = ["t", "sin1", "cos1", "holiday_week"]
        if n_harm >= 2:
            cols += ["sin2", "cos2"]
        if use_promo:
            cols += ["promo"]
        if use_lag1:
            hist["lag1"] = hist["sales"].shift(1)
            cols += ["lag1"]
            hist = hist.dropna(subset=["lag1"])
        X = hist[cols].values
        y = hist["sales"].values
        model = LinearRegression().fit(X, y)
        fut = g.loc[g["t"].isin([T + h for h in horizons])].copy()
        fut = fut.set_index(fut["t"] - T)
        if use_promo:
            fut["promo"] = 0  # no future promo calendar assumed
        if use_lag1:
            last_known = g.loc[g["t"] == T, "sales"].values[0]
            fut["lag1"] = last_known  # carry last observed value forward as "momentum"
        Xf = fut[cols].values
        preds = np.clip(model.predict(Xf), 0, None)
        for h, p in zip(fut.index, preds):
            rows.append((sid, h, p))
    return pd.DataFrame(rows, columns=["store_id", "h", "forecast"])


def run_variant(name, **kwargs):
    preds = pd.concat([forecast_harmonic_v2(feat, T, horizons, **kwargs).assign(T=T) for T in origins], ignore_index=True)
    preds["target_t"] = preds["T"] + preds["h"]
    preds["actual"] = preds.apply(lambda r: actual_lookup.get((r["store_id"], r["target_t"]), np.nan), axis=1)
    preds = preds.dropna(subset=["actual"])
    m = metrics(preds)
    print(f"{name}: WAPE={m['WAPE']:.4f} MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} BIAS={m['BIAS']:.1f}")
    return preds


print("baseline (2 harmonics, no promo, no lag1):")
run_variant("C_base", use_promo=False, n_harm=2, use_lag1=False)

print("\nwith promo history (assume 0 at forecast):")
run_variant("C_promo", use_promo=True, n_harm=2, use_lag1=False)

print("\nwith 1 harmonic only:")
run_variant("C_1harm", use_promo=False, n_harm=1, use_lag1=False)

print("\nwith lag1 momentum term:")
run_variant("C_lag1", use_promo=False, n_harm=2, use_lag1=True)

print("\nwith promo + lag1:")
run_variant("C_promo_lag1", use_promo=True, n_harm=2, use_lag1=True)

print("\nDone.")
