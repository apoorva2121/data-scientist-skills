"""
common.py - shared data cleaning, feature construction and model code.

Used by 02_backtest.py (model comparison) and 03_forecast.py (final forecast).

Cleaning decisions (see report.md for rationale):
  * exact duplicate rows are dropped (4 pairs)
  * sales == -1 is treated as missing (5 rows, stores S10/S12)
  * every store is reindexed onto the full weekly calendar (S03 has 3 missing weeks)
  * holiday_week is a store-invariant calendar, so gaps are filled from other stores
  * promo for missing weeks is assumed 0
  * S19 (intermittent, ~60% zero weeks) is excluded from model fitting and from headline
    accuracy; it gets a simple intermittent-demand forecast (mean of last 52 weeks)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
H = 8                      # forecast horizon (weeks)
SEASON = 52
INTERMITTENT_STORES = ["S19"]
LEVEL_WIN = 13             # weeks used to define "current level" of a store


# ----------------------------------------------------------------------------- data
def load_clean():
    df = pd.read_csv(DATA, parse_dates=["week_start"])
    n0 = len(df)
    df = df.drop_duplicates()
    assert not df.duplicated(["store_id", "week_start"]).any(), "non-identical duplicate rows remain"
    df.loc[df["sales"] < 0, "sales"] = np.nan          # -1 sentinel -> missing

    weeks = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="7D")
    stores = sorted(df["store_id"].unique())
    idx = pd.MultiIndex.from_product([stores, weeks], names=["store_id", "week_start"])
    df = df.set_index(["store_id", "week_start"]).reindex(idx).reset_index()

    df["holiday_week"] = df.groupby("week_start")["holiday_week"].transform("max").astype(int)
    df["promo"] = df["promo"].fillna(0).astype(int)
    df["t"] = ((df["week_start"] - weeks[0]).dt.days // 7).astype(int)
    df["woy"] = df["week_start"].dt.isocalendar().week.astype(int)
    df = df.sort_values(["store_id", "t"]).reset_index(drop=True)
    info = dict(rows_raw=n0, rows_after_dedupe=n0 - 4, rows_full_grid=len(df),
                n_missing_sales=int(df["sales"].isna().sum()), n_weeks=len(weeks), stores=stores)
    return df, weeks, info


def fill_sales(train):
    """Linear interpolation of missing sales within a store, using only rows passed in
    (call on the training slice so test values never leak into the fill)."""
    train = train.copy()
    train["sales_f"] = (train.groupby("store_id")["sales"]
                        .transform(lambda s: s.interpolate(limit_direction="both")))
    return train


# ----------------------------------------------------------------------------- baselines
def _per_store_arrays(train):
    out = {}
    for sid, g in train.groupby("store_id"):
        out[sid] = g["sales_f"].to_numpy()
    return out


def baseline_predict(kind, train, origin, future):
    """kind in {naive, mean8, snaive, snaive_drift}. `future` has store_id, t, h rows."""
    arrs = _per_store_arrays(train)
    preds = []
    for _, r in future.iterrows():
        y = arrs[r.store_id]
        tw = int(r.t)
        if kind == "naive":
            p = y[-1]
        elif kind == "mean8":
            p = y[-8:].mean()
        elif kind == "snaive":
            p = y[tw - SEASON] if tw - SEASON >= 0 else y[-8:].mean()
        elif kind == "snaive_drift":
            if tw - SEASON >= 0 and origin - SEASON - LEVEL_WIN + 1 >= 0:
                lvl_now = y[origin - LEVEL_WIN + 1: origin + 1].mean()
                lvl_ly = y[origin - SEASON - LEVEL_WIN + 1: origin - SEASON + 1].mean()
                p = y[tw - SEASON] * lvl_now / lvl_ly
            elif tw - SEASON >= 0:
                p = y[tw - SEASON]
            else:
                p = y[-8:].mean()
        else:
            raise ValueError(kind)
        preds.append(p)
    return np.array(preds)


# ----------------------------------------------------------------------------- Fourier regression
def _fourier_X(t, promo, hol, K, promo_prev=None):
    cols = [np.ones_like(t, dtype=float), t.astype(float) / SEASON]
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / SEASON))
        cols.append(np.cos(2 * np.pi * k * t / SEASON))
    cols.append(promo.astype(float))
    cols.append(hol.astype(float))
    if promo_prev is not None:
        cols.append(promo_prev.astype(float))
    return np.column_stack(cols)


def fourier_predict(train, future, K=2, ar1=False, ridge=1e-3, use_promo_prev=False):
    """Per-store OLS on log(sales): trend + Fourier(K) + promo + holiday (+ AR(1) residual)."""
    preds = np.zeros(len(future))
    fut = future.reset_index(drop=True)
    for sid, g in train.groupby("store_id"):
        m = fut["store_id"] == sid
        if not m.any():
            continue
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        y = np.log(g["sales_f"].to_numpy())
        pp = g["promo"].shift(1).fillna(0).to_numpy() if use_promo_prev else None
        X = _fourier_X(t, g["promo"].to_numpy(), g["holiday_week"].to_numpy(), K, pp)
        # ridge-stabilised least squares (intercept not penalised in practice: tiny ridge)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ y)
        resid = y - X @ beta
        f = fut[m]
        ppf = None
        if use_promo_prev:
            # promo of the week before the target: for h=1 it's the last training week
            full = pd.concat([g[["t", "promo"]], f[["t", "promo"]]]).drop_duplicates("t").set_index("t")["promo"]
            ppf = full.reindex(f["t"].to_numpy() - 1).fillna(0).to_numpy()
        Xf = _fourier_X(f["t"].to_numpy(), f["promo"].to_numpy(), f["holiday_week"].to_numpy(), K, ppf)
        yhat = Xf @ beta
        if ar1 and len(resid) > 10:
            phi = np.corrcoef(resid[:-1], resid[1:])[0, 1]
            phi = float(np.clip(phi, 0, 0.95))
            h = f["h"].to_numpy()
            yhat = yhat + (phi ** h) * resid[-1]
        preds[m.to_numpy()] = np.exp(yhat)
    return preds


# ----------------------------------------------------------------------------- global LightGBM (direct multi-horizon)
FEATS = ["h", "store_code", "log_level", "r1", "r4", "r8", "trend13", "promo_rate13",
         "yoy_growth", "seas_ratio_ly", "seas_ratio_ly_m1", "seas_ratio_ly_p1",
         "woy_sin1", "woy_cos1", "woy_sin2", "woy_cos2", "woy",
         "promo_tgt", "hol_tgt", "promo_prev", "inv_ratio"]


def _store_series(df_all, sid):
    g = df_all[df_all["store_id"] == sid].sort_values("t")
    return g


def build_rows(panel, origins, covar, horizons=range(1, H + 1), store_codes=None):
    """
    panel : DataFrame for ONE store with columns t, sales_f (filled, may be NaN beyond origin
            for the forecast case), promo, holiday_week, inventory_end - training history.
    origins: iterable of origin t values (last observed week).
    covar : DataFrame with t, promo, holiday_week covering all target weeks (known calendar).
    Returns feature rows for every (origin, h).
    """
    p = panel.sort_values("t").set_index("t")
    y = p["sales_f"]
    ly = np.log(y)
    lvl = y.rolling(LEVEL_WIN, min_periods=8).mean()
    m1, m4, m8 = y.rolling(1).mean(), y.rolling(4, min_periods=2).mean(), y.rolling(8, min_periods=4).mean()
    prev13 = y.shift(LEVEL_WIN).rolling(LEVEL_WIN, min_periods=8).mean()
    prate13 = p["promo"].rolling(LEVEL_WIN, min_periods=8).mean()
    inv = p["inventory_end"]
    cv = covar.set_index("t")
    rows = []
    for o in origins:
        if o not in lvl.index or np.isnan(lvl.get(o, np.nan)):
            continue
        L = lvl[o]
        base = dict(origin=o, log_level=np.log(L),
                    r1=np.log(m1[o] / L), r4=np.log(m4[o] / L), r8=np.log(m8[o] / L),
                    trend13=np.log(L / prev13[o]) if o in prev13.index and not np.isnan(prev13[o]) else np.nan,
                    promo_rate13=prate13[o],
                    inv_ratio=np.log(inv[o] / L) if o in inv.index and not np.isnan(inv[o]) else np.nan)
        L52 = lvl.get(o - SEASON, np.nan)
        base["yoy_growth"] = np.log(L / L52) if not np.isnan(L52) else np.nan
        for h in horizons:
            tw = o + h
            if tw not in cv.index:
                continue
            r = dict(base, h=h, t=tw, woy=int(cv.loc[tw, "woy"]),
                     promo_tgt=int(cv.loc[tw, "promo"]), hol_tgt=int(cv.loc[tw, "holiday_week"]),
                     promo_prev=int(cv.loc[tw - 1, "promo"]) if (tw - 1) in cv.index else 0)
            for off, name in [(0, "seas_ratio_ly"), (-1, "seas_ratio_ly_m1"), (1, "seas_ratio_ly_p1")]:
                tl = tw - SEASON + off
                r[name] = np.log(y[tl] / L52) if (tl in y.index and not np.isnan(y[tl]) and not np.isnan(L52)) else np.nan
            r["y"] = np.log(y[tw] / L) if (tw in y.index and not np.isnan(y[tw])) else np.nan
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["woy_sin1"] = np.sin(2 * np.pi * out["woy"] / SEASON)
        out["woy_cos1"] = np.cos(2 * np.pi * out["woy"] / SEASON)
        out["woy_sin2"] = np.sin(4 * np.pi * out["woy"] / SEASON)
        out["woy_cos2"] = np.cos(4 * np.pi * out["woy"] / SEASON)
    return out


LGB_PARAMS = dict(objective="l2", n_estimators=500, learning_rate=0.03, num_leaves=15,
                  min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=2.0, verbose=-1, random_state=42)


def lgbm_fit(train, covar_all, store_codes, params=None, feats=None, min_origin=LEVEL_WIN - 1):
    """Build training rows from every (store, origin, h) fully inside `train` and fit."""
    params = params or LGB_PARAMS
    feats = feats or FEATS
    T = int(train["t"].max())
    frames = []
    for sid, g in train.groupby("store_id"):
        cov = covar_all[(covar_all["store_id"] == sid) & (covar_all["t"] <= T)][["t", "promo", "holiday_week", "woy"]]
        rows = build_rows(g, range(min_origin, T), cov)
        if len(rows):
            rows["store_id"] = sid
            frames.append(rows)
    X = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    X["store_code"] = X["store_id"].map(store_codes)
    model = lgb.LGBMRegressor(**params)
    model.fit(X[feats], X["y"], categorical_feature=["store_code"])
    return model, X


def lgbm_predict(model, train, covar_all, origin, future, store_codes, feats=None):
    feats = feats or FEATS
    fut = future.reset_index(drop=True)
    preds = np.full(len(fut), np.nan)
    for sid, g in train.groupby("store_id"):
        m = (fut["store_id"] == sid).to_numpy()
        if not m.any():
            continue
        cov = covar_all[covar_all["store_id"] == sid][["t", "promo", "holiday_week", "woy"]]
        # future covariates come from `future` (they may be scenario-specific)
        cov = pd.concat([cov[cov["t"] <= origin], fut.loc[m, ["t", "promo", "holiday_week", "woy"]]]).drop_duplicates("t")
        rows = build_rows(g, [origin], cov)
        rows["store_code"] = store_codes[sid]
        rows = rows.set_index("h").reindex(fut.loc[m, "h"].to_numpy()).reset_index()
        yhat = model.predict(rows[feats])
        preds[m] = np.exp(rows["log_level"].to_numpy() + yhat)
    return preds


# ----------------------------------------------------------------------------- metrics
def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[ok], p[ok]
    e = p - a
    return dict(n=len(a), WAPE=np.abs(e).sum() / a.sum(), MAE=np.abs(e).mean(),
                RMSE=np.sqrt((e ** 2).mean()), bias_pct=e.sum() / a.sum(),
                sMAPE=np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(p))))
