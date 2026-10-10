"""Stage 9 helpers: the one-time test score with an interval, threshold, calibration, slices, sanity.

    from final_eval import (bootstrap_ci, capacity_threshold, calibration_table, slice_table,
                            shuffled_label_check, interval_coverage)

All functions take plain arrays (y true, s = score or prediction) so they work for any model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone


def bootstrap_ci(y, s, metric: Callable, n_boot: int = 2000, alpha: float = 0.05, stratified: bool = True,
                 seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval of metric(y, s) over test rows.

    stratified=True resamples positives and negatives separately (binary targets), so rare
    classes are never absent from a resample.
    """
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    point = float(metric(y, s))
    if stratified and set(np.unique(y)) <= {0, 1}:
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n_boot))
    else:
        draws = (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
    stats = np.array([metric(y[i], s[i]) for i in draws])
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def capacity_threshold(s, frac: float) -> float:
    """Score threshold that selects the top `frac` of rows (e.g. call capacity of 10%)."""
    return float(np.quantile(np.asarray(s), 1 - frac))


def confusion_at(y, s, threshold: float) -> dict:
    y, pred = np.asarray(y), (np.asarray(s) >= threshold).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "flagged_share": (tp + fp) / len(y)}


def calibration_table(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Quantile-binned reliability table, expected calibration error (ECE) and Brier score."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = pd.qcut(pd.Series(p).rank(method="first"), q=n_bins, labels=False)
    tab = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                                        observed=("y", "mean"))
    ece = float((tab["n"] * (tab["mean_pred"] - tab["observed"]).abs()).sum() / tab["n"].sum())
    brier = float(np.mean((p - y) ** 2))
    return tab.round(4), ece, brier


def slice_table(slices: pd.DataFrame, y, s, metric: Callable, min_n: int = 30, n_boot: int = 500) -> pd.DataFrame:
    """Metric with a bootstrap interval for every level of every slice column (levels under min_n skipped)."""
    y, s = np.asarray(y), np.asarray(s)
    rows = []
    for col in slices.columns:
        for level, idx in slices.groupby(col, dropna=False).indices.items():
            if len(idx) < min_n or len(np.unique(y[idx])) < 2 and set(np.unique(y)) <= {0, 1}:
                continue
            point, lo, hi = bootstrap_ci(y[idx], s[idx], metric, n_boot=n_boot)
            rows.append({"slice": col, "level": level, "n": len(idx), "metric": round(point, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    return pd.DataFrame(rows).sort_values(["slice", "metric"]).reset_index(drop=True)


def _fresh(estimator):
    """clone(), falling back to deepcopy for estimators that clone() rejects (e.g. CatBoost with cat_features)."""
    import copy
    try:
        return clone(estimator)
    except RuntimeError:
        return copy.deepcopy(estimator)


def shuffled_label_check(pipe, X: pd.DataFrame, y, folds, scorer, seed: int = 0) -> tuple[float, float]:
    """CV score with the real labels vs with shuffled labels. Shuffled must fall to the trivial baseline;
    if it does not, something outside the labels (row order, IDs, a leak via preprocessing) carries signal."""
    y = np.asarray(y)
    y_shuf = np.random.default_rng(seed).permutation(y)
    real, fake = [], []
    for tr, va in folds:
        real.append(scorer(_fresh(pipe).fit(X.iloc[tr], y[tr]), X.iloc[va], y[va]))
        fake.append(scorer(_fresh(pipe).fit(X.iloc[tr], y_shuf[tr]), X.iloc[va], y_shuf[va]))
    return float(np.mean(real)), float(np.mean(fake))


def interval_coverage(y, lower, upper) -> dict:
    y, lower, upper = map(np.asarray, (y, lower, upper))
    inside = (y >= lower) & (y <= upper)
    return {"coverage": float(inside.mean()), "mean_width": float(np.mean(upper - lower)),
            "below": float((y < lower).mean()), "above": float((y > upper).mean())}
