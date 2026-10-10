"""Rolling-origin backtesting for panels of time series (pandas + numpy only).

    from ts_backtest import regularize, rolling_origins, backtest, seasonal_naive, summarize
    panel, gaps = regularize(df, id_col="series_id", time_col="date", target="demand", freq="W-MON")
    origins = rolling_origins(panel["ds"], horizon=8, n_origins=6, step=4, holdout_last=16)
    res = backtest(panel, seasonal_naive(season=52), horizon=8, origins=origins, season=52)
    print(summarize(res))                  # overall, by horizon, by series

Panel format used throughout: columns unique_id, ds, y (+ any covariates), one row per series per
period, on a regular calendar. A forecaster is any function
    f(history: DataFrame[unique_id, ds, y, ...], future: DataFrame[unique_id, ds, h, ...]) -> array of yhat
aligned with `future` rows. `future` carries the step ahead `h` (1..horizon) and only covariates known
in advance (you choose which columns to pass via `known_future`), never y.

    breaks = break_report(panel, window=13, ratio=1.3, season=52)   # level shifts per series, before modelling
    table = paired_compare({"snaive": res_a, "lgbm": res_b})        # Stage 8: paired by origin, 1-SE keep

Metrics are scale-free so series of different sizes can be averaged:
  MASE  = MAE / in-sample MAE of the seasonal naive forecast on that series' training history
  WAPE  = sum |error| / sum |actual|
  sMAPE = mean 2|e| / (|y| + |yhat|)
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd


def regularize(df: pd.DataFrame, id_col: str, time_col: str, target: str, freq: str):
    """Reindex every series onto a complete calendar. Returns (panel, gap_report).

    Rows created for missing periods get y = NaN; the report lists them so the analyst decides
    (zero = no activity, or genuinely missing = fill or leave for a NaN-aware model).
    """
    d = df.rename(columns={id_col: "unique_id", time_col: "ds", target: "y"}).copy()
    d["ds"] = pd.to_datetime(d["ds"])
    if d.duplicated(["unique_id", "ds"]).any():
        raise ValueError("duplicate (series, period) rows: resolve them at Stage 1 first")
    d["_present"] = True
    full = []
    for uid, g in d.groupby("unique_id", sort=False):
        cal = pd.date_range(g["ds"].min(), g["ds"].max(), freq=freq)
        full.append(g.set_index("ds").reindex(cal).rename_axis("ds").reset_index().assign(unique_id=uid))
    panel = pd.concat(full, ignore_index=True)
    missing = panel["y"].isna()
    gaps = panel.loc[missing, ["unique_id", "ds"]].assign(
        gap_type=np.where(panel.loc[missing, "_present"].eq(True), "row present, y missing", "period absent from data"))
    return panel.drop(columns="_present"), gaps.reset_index(drop=True)


def break_report(panel: pd.DataFrame, window: int = 13, ratio: float = 1.3, min_zero_share: float = 0.3,
                 season: int | None = None) -> pd.DataFrame:
    """Per series: the largest level change between two `window`-period medians, and the zero share.

    With `season` given (52 for weekly data with a yearly cycle), each window is compared with the
    same weeks one season earlier, so seasonal swings cancel and only level changes and trend remain.
    Without it, consecutive windows are compared, which confuses seasonality with shifts.
    A level shift (ratio >= `ratio` or <= 1/`ratio`) means the history before the break misleads
    most models: train on post-break data, add a regime flag, or prefer adaptive models.
    A zero share >= `min_zero_share` marks intermittent demand: use intermittent methods and judge it
    by MASE or absolute error, since percentage metrics such as WAPE explode near zero.
    """
    rows = []
    for uid, g in panel.sort_values("ds").groupby("unique_id"):
        y = g["y"].to_numpy(dtype=float)
        best, at = 1.0, None
        lag = season if season else window
        for i in range(lag, len(y) - window + 1):
            before, after = np.nanmedian(y[i - lag:i - lag + window]), np.nanmedian(y[i:i + window])
            if before > 0 and after > 0:
                r = after / before
                if abs(np.log(r)) > abs(np.log(best)):
                    best, at = r, g["ds"].iloc[i]
        zero = float(np.nanmean(y == 0))
        flags = []
        if best >= ratio or best <= 1 / ratio:
            flags.append("LEVEL_SHIFT")
        if zero >= min_zero_share:
            flags.append("INTERMITTENT")
        rows.append({"unique_id": uid, "max_level_ratio": round(best, 2),
                     "shift_at": at if flags and "LEVEL_SHIFT" in flags else None,
                     "zero_share": round(zero, 2), "issues": " | ".join(flags)})
    return pd.DataFrame(rows).sort_values("max_level_ratio", key=lambda s: -np.abs(np.log(s))).reset_index(drop=True)


def rolling_origins(ds: pd.Series, horizon: int, n_origins: int, step: int, holdout_last: int = 0) -> list:
    """Last-observed dates for each backtest origin, newest last, leaving `holdout_last` periods untouched."""
    dates = np.sort(pd.to_datetime(pd.Series(ds)).unique())
    usable = dates[: len(dates) - holdout_last] if holdout_last else dates
    last_origin = len(usable) - horizon - 1
    idx = [last_origin - k * step for k in range(n_origins)][::-1]
    if idx[0] < 1:
        raise ValueError("not enough history for these origins; reduce n_origins or step")
    return [pd.Timestamp(usable[i]) for i in idx]


def seasonal_naive(season: int) -> Callable:
    """y_hat(t + h) = y(t + h - season * k), the most recent same-season value."""

    def f(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        out = np.full(len(future), np.nan)
        for uid, fut in future.groupby("unique_id", sort=False):
            hist = history[history["unique_id"] == uid].sort_values("ds")["y"].to_numpy()
            n = len(hist)
            for j, i in enumerate(fut.index):
                h = j + 1
                lag = season * int(np.ceil(h / season))
                out[future.index.get_loc(i)] = hist[n - lag + h - 1] if n - lag + h - 1 >= 0 else hist[-1]
        return out

    return f


def naive_last() -> Callable:
    def f(history, future):
        last = history.sort_values("ds").groupby("unique_id")["y"].last()
        return future["unique_id"].map(last).to_numpy(dtype=float)

    return f


def _mase_scale(y: np.ndarray, season: int) -> float:
    y = y[~np.isnan(y)]
    if len(y) <= season:
        return np.nan
    d = np.abs(y[season:] - y[:-season])
    return float(np.mean(d)) if np.mean(d) > 0 else np.nan


def backtest(panel: pd.DataFrame, forecaster: Callable, horizon: int, origins: Iterable, season: int,
             known_future: Iterable[str] = ()) -> pd.DataFrame:
    """Fit-on-history, forecast-the-next-`horizon` at every origin. Returns one row per (series, origin, h)."""
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    cols = ["unique_id", "ds", "h", *known_future]  # h (1..horizon) lets direct multi-horizon models use it
    rows = []
    for origin in origins:
        hist = panel[panel["ds"] <= origin]
        fut = (panel[panel["ds"] > origin].groupby("unique_id", sort=False).head(horizon)).copy()
        fut["h"] = fut.groupby("unique_id").cumcount() + 1
        yhat = np.asarray(forecaster(hist.copy(), fut[cols].copy()), dtype=float)
        scale = hist.groupby("unique_id")["y"].apply(lambda s: _mase_scale(s.to_numpy(), season))
        rows.append(pd.DataFrame({"unique_id": fut["unique_id"].to_numpy(), "origin": origin, "h": fut["h"].to_numpy(),
                                  "ds": fut["ds"].to_numpy(), "y": fut["y"].to_numpy(), "yhat": yhat,
                                  "scale": fut["unique_id"].map(scale).to_numpy()}))
    res = pd.concat(rows, ignore_index=True)
    return res[res["y"].notna()]


def _metrics(g: pd.DataFrame) -> pd.Series:
    e = (g["y"] - g["yhat"]).abs()
    return pd.Series({"MASE": (e / g["scale"]).mean(), "WAPE": e.sum() / g["y"].abs().sum(),
                      "sMAPE": (2 * e / (g["y"].abs() + g["yhat"].abs()).replace(0, np.nan)).mean(), "n": len(g)})


def paired_compare(results: dict, metric: str = "MASE") -> pd.DataFrame:
    """Compare backtest results of several forecasters on the SAME origins (Stage 8 for forecasting).

    results: {name: backtest(...) output}. Per origin, the metric is computed for each candidate;
    differences to the best are paired by origin (origin difficulty cancels). keep = within one
    standard error of the paired difference. With origin steps shorter than the horizon, forecast
    windows overlap and the SE is optimistic; prefer step >= horizon for the final comparison.
    """
    per_origin = pd.DataFrame({name: _by(res, "origin")[metric] for name, res in results.items()})
    means = per_origin.mean()
    lower_is_better = metric in ("MASE", "WAPE", "sMAPE")
    best = means.idxmin() if lower_is_better else means.idxmax()
    rows = []
    for name in per_origin.columns:
        d = per_origin[name] - per_origin[best]
        d = -d if lower_is_better else d  # positive = better than best (never, by construction)
        n = int(d.notna().sum())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append({"name": name, metric: round(float(means[name]), 4), "sd_across_origins": round(float(per_origin[name].std(ddof=1)), 4),
                     "diff_vs_best": round(float(d.mean()), 4), "se_paired": round(se, 4), "n_origins": n,
                     "keep": bool(name == best or (np.isfinite(se) and d.mean() >= -se))})
    return pd.DataFrame(rows).sort_values(metric, ascending=lower_is_better).reset_index(drop=True)


def _by(res: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame({k: _metrics(g) for k, g in res.groupby(col)}).T.rename_axis(col)


def summarize(res: pd.DataFrame) -> dict:
    """Overall, by horizon, by series and by origin (the spread across origins is the noise floor)."""
    return {"overall": _metrics(res), "by_h": _by(res, "h"), "by_series": _by(res, "unique_id"),
            "by_origin": _by(res, "origin")}
