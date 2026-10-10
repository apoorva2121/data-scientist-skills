"""Stage 4 diagnosis: evidence about WHY values are missing, per column, on training rows only.

Usage:
    python missing_profile.py TRAIN.csv --target TARGET --task binary|multiclass|regression
        [--exclude id_cols] [--time DATE_COL] [--out missing_profile.csv]

Per column with any missing value:
  pct_missing           share of rows missing
  predictable_from_obs  CV AUC of predicting "is missing" from the OTHER columns (0.5 = unrelated)
  target_signal         missing-indicator vs target: AUC-type for classification, |Spearman| for regression
  co_missing_with       columns whose missing masks overlap strongly (Jaccard >= 0.8)
  trend_by_period       missing share in the first vs last quarter of time (collection change)
  hint                  structural? / MCAR-like / MAR-like / informative (see SKILL.md for what to do)

Hints are evidence summaries, not verdicts. MNAR cannot be established from the data alone:
if missingness depends on the unobserved value itself, only domain knowledge can say so.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")


def _auc(ind: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(ind)) < 2:
        return np.nan
    a = roc_auc_score(y, ind)
    return max(a, 1 - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    ap.add_argument("--exclude", default="")
    ap.add_argument("--time", default="")
    ap.add_argument("--out", default="missing_profile.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    exclude = {c.strip() for c in a.exclude.split(",") if c.strip()} | {a.target}
    feats = [c for c in df.columns if c not in exclude and c != a.time]
    y = df[a.target]
    miss = df[feats].isna()
    cols = [c for c in feats if miss[c].any()]
    if not cols:
        print("no missing values in feature columns")
        return

    X_obs = df[feats].copy()
    for c in X_obs.columns:
        if not pd.api.types.is_numeric_dtype(X_obs[c]):
            X_obs[c] = X_obs[c].astype("category")

    rows = []
    for c in cols:
        m = miss[c].to_numpy().astype(int)
        others = X_obs.drop(columns=[c])
        pred = np.nan
        if 20 <= m.sum() <= len(m) - 20:
            clf = HistGradientBoostingClassifier(max_iter=100, categorical_features="from_dtype", random_state=0)
            p = cross_val_predict(clf, others, m, cv=5, method="predict_proba")[:, 1]
            pred = roc_auc_score(m, p)
        if a.task == "regression":
            sig = abs(spearmanr(m, y.astype(float)).correlation)
        elif a.task == "binary":
            sig = _auc(m, pd.factorize(y, sort=True)[0])
        else:
            codes = pd.factorize(y, sort=True)[0]
            sig = np.nanmean([_auc(m, (codes == k).astype(int)) for k in np.unique(codes)])
        co = [o for o in cols if o != c and (miss[c] & miss[o]).sum() / max((miss[c] | miss[o]).sum(), 1) >= 0.8]
        trend = ""
        if a.time:
            t = pd.to_datetime(df[a.time])
            q1, q4 = t.quantile(0.25), t.quantile(0.75)
            trend = f"{miss.loc[t <= q1, c].mean():.2f} -> {miss.loc[t >= q4, c].mean():.2f}"

        informative = (sig >= 0.55) if a.task != "regression" else (sig > max(3 / np.sqrt(len(df)), 0.05))
        if miss[c].mean() > 0 and co and informative is False and not np.isnan(pred) and pred > 0.9:
            hint = "structural? (missing together with related columns, predictable from them)"
        elif informative:
            hint = "informative: keep a missing indicator or native NaN"
        elif np.isnan(pred) or pred < 0.6:
            hint = "MCAR-like: simple imputation adequate"
        else:
            hint = "MAR-like: impute from other columns (KNN / iterative), add indicator"
        rows.append({"column": c, "pct_missing": round(100 * miss[c].mean(), 2),
                     "predictable_from_obs": round(pred, 3) if not np.isnan(pred) else np.nan,
                     "target_signal": round(float(sig), 3), "co_missing_with": ",".join(co),
                     "trend_by_period": trend, "hint": hint})

    res = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    res.to_csv(a.out, index=False)
    print(f"rows={len(df)}; rows with any missing feature={int(miss.any(axis=1).sum())} "
          f"({100 * miss.any(axis=1).mean():.1f}%)")
    print(res.to_string(index=False))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
