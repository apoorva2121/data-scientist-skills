"""Stage 3 leak screen: univariate predictive strength of every column, on TRAINING data only.

Usage:
    python leak_screen.py TRAIN.csv --target TARGET --task binary|multiclass|regression
        [--exclude col1,col2] [--groups GROUP_COL] [--out leak_screen.csv]

Run after stateless cleaning (sentinels already turned into missing values) and after the test set
has been carved off. A flag is a question to answer, not a verdict: every flagged column needs a
record-timing answer ("when is this value written, relative to the prediction moment?").

Scores
  binary      max(AUC, 1-AUC) of the column alone        0.5 = no signal, 1.0 = perfect
  multiclass  mean one-vs-rest score as above
  regression  |Spearman rho| of the column alone         0 = no signal, 1.0 = perfect
Categorical columns are scored through out-of-fold target means (grouped folds if --groups).

Flags
  NEAR_PERFECT               score >= 0.90 (classification) or >= 0.95 (regression)
  DOMINANT                   robust z-score of the column's score vs the others >= 5
  INFORMATIVE_MISSINGNESS    the missing-indicator alone is significant: binary AUC > 0.5 + 3 SE (Hanley-McNeil)
                             and >= 0.53; multiclass >= 0.55; regression |rho| > max(3/sqrt(n), 0.05)
  DERIVED_FROM_TARGET?       regression only: log(target) ~ log(col) [+ log(other col)] has R^2 >= 0.98 on the
                             best-fitting 95% of rows (trimmed, so a few corrupted rows cannot hide the relation)
  ROW_ORDER_SIGNAL           target is predictable from file order (sorted export, or leaky IDs)
"""
from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")


def _binary_score(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~pd.isna(x)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or len(np.unique(x[ok])) < 2:
        return np.nan
    auc = roc_auc_score(y[ok], x[ok].astype(float))
    return max(auc, 1 - auc)


def _reg_score(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~pd.isna(x)
    if ok.sum() < 20 or len(np.unique(x[ok])) < 2:
        return np.nan
    return abs(spearmanr(x[ok].astype(float), y[ok]).correlation)


def _score(x: np.ndarray, y: np.ndarray, task: str, classes) -> float:
    if task == "binary":
        return _binary_score(x, y)
    if task == "regression":
        return _reg_score(x, y)
    vals = [_binary_score(x, (y == c).astype(int)) for c in classes]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def _oof_target_mean(col: pd.Series, y_num: np.ndarray, splitter, groups) -> np.ndarray:
    cat = col.astype("string").fillna("__NA__").to_numpy()
    out = np.full(len(cat), np.nan)
    for tr, va in splitter.split(cat, y_num, groups):
        means = pd.Series(y_num[tr]).groupby(cat[tr]).mean()
        out[va] = pd.Series(cat[va]).map(means).fillna(y_num[tr].mean()).to_numpy()
    return out


def _trimmed_r2(X: np.ndarray, t: np.ndarray, keep: float = 0.95, rounds: int = 3) -> float:
    """R^2 of a least-squares fit refitted on the best-fitting `keep` share of rows.

    A column derived from the target (unit_price = revenue / quantity, target revenue) is an exact relation on almost
    every row, but a few corrupted rows destroy an ordinary R^2. Trimming finds the exact core.
    """
    idx = np.arange(len(t))
    for _ in range(rounds):
        beta = np.linalg.lstsq(X[idx], t[idx], rcond=None)[0]
        resid = np.abs(t - X @ beta)
        idx = np.argsort(resid)[: max(10, int(keep * len(t)))]
    beta = np.linalg.lstsq(X[idx], t[idx], rcond=None)[0]
    return float(1 - np.var(t[idx] - X[idx] @ beta) / np.var(t[idx]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    ap.add_argument("--exclude", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--out", default="leak_screen.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df[df[a.target].notna()].reset_index(drop=True)
    exclude = {c.strip() for c in a.exclude.split(",") if c.strip()} | {a.target}
    groups = df[a.groups].to_numpy() if a.groups else None
    splitter = GroupKFold(n_splits=5) if a.groups else KFold(n_splits=5, shuffle=True, random_state=0)

    y_raw = df[a.target]
    classes = None
    if a.task == "binary":
        y = pd.factorize(y_raw, sort=True)[0]
        y_num = y.astype(float)
    elif a.task == "multiclass":
        y, classes = pd.factorize(y_raw, sort=True)
        classes = range(len(classes))
        y_num = y.astype(float)  # only used for OOF means per class below
    else:
        y = y_raw.astype(float).to_numpy()
        y_num = y

    rows = []
    for c in df.columns:
        if c in exclude:
            continue
        col = df[c]
        numeric = pd.api.types.is_numeric_dtype(col)
        if numeric:
            x = col.to_numpy(dtype=float)
            kind = "numeric"
        elif a.task == "multiclass":
            enc = [_oof_target_mean(col, (y == k).astype(float), splitter, groups) for k in classes]
            s = np.nanmean([_binary_score(e, (y == k).astype(int)) for k, e in zip(classes, enc)])
            rows.append({"column": c, "kind": "categorical", "score": s, "missing_pct": round(100 * col.isna().mean(), 2)})
            continue
        else:
            x = _oof_target_mean(col, y_num, splitter, groups)
            kind = "categorical"
        row = {"column": c, "kind": kind, "score": _score(x, y, a.task, classes),
               "missing_pct": round(100 * col.isna().mean(), 2)}
        if col.isna().any() and col.notna().any():
            row["missing_indicator_score"] = _score(col.isna().astype(int).to_numpy(), y, a.task, classes)
        rows.append(row)

    rows.append({"column": "__row_order__", "kind": "diagnostic",
                 "score": _score(np.arange(len(df), dtype=float), y, a.task, classes), "missing_pct": 0.0})
    res = pd.DataFrame(rows)

    near = 0.95 if a.task == "regression" else 0.90
    base = 0.0 if a.task == "regression" else 0.5
    n = len(df)
    if a.task == "binary":
        n1 = int((y == 1).sum()); n0 = n - n1

        def miss_significant(s: float) -> bool:
            # Hanley & McNeil (1982) standard error of an AUC; require > 3 SE above 0.5 and a practical 0.53.
            q1, q2 = s / (2 - s), 2 * s * s / (1 + s)
            se = np.sqrt((s * (1 - s) + (n1 - 1) * (q1 - s * s) + (n0 - 1) * (q2 - s * s)) / (n1 * n0))
            return s - 0.5 > 3 * se and s >= 0.53
    elif a.task == "multiclass":
        def miss_significant(s: float) -> bool:
            return s >= 0.55
    else:
        def miss_significant(s: float) -> bool:
            return s > max(3 / np.sqrt(n), 0.05)
    feat = res[res["kind"] != "diagnostic"]["score"].dropna()
    med = feat.median()
    mad = 1.4826 * (feat - med).abs().median()

    def flags(r) -> str:
        f = []
        s = r["score"]
        if r["kind"] == "diagnostic":
            if pd.notna(s) and s - base >= (0.10 if a.task != "regression" else 0.20):
                f.append("ROW_ORDER_SIGNAL")
            return " | ".join(f)
        if pd.notna(s) and s >= near:
            f.append("NEAR_PERFECT")
        if pd.notna(s) and mad > 0 and (s - med) / mad >= 5:
            f.append("DOMINANT")
        mi = r.get("missing_indicator_score", np.nan)
        if pd.notna(mi) and miss_significant(mi):
            f.append("INFORMATIVE_MISSINGNESS")
        return " | ".join(f)

    res["flags"] = res.apply(flags, axis=1)

    if a.task == "regression" and (y > 0).all():
        ly = np.log(y)
        pos = [c for c in res[res["kind"] == "numeric"].sort_values("score", ascending=False)["column"].head(12)
               if (df[c].dropna() > 0).all()]
        derived = {}
        for c in pos:
            ok = df[c].notna().to_numpy()
            X = np.column_stack([np.ones(ok.sum()), np.log(df.loc[ok, c])])
            r2 = _trimmed_r2(X, ly[ok])
            if r2 >= 0.98:
                derived[c] = f"alone trimmed R2={r2:.3f}"
        for c1, c2 in itertools.combinations(pos, 2):
            ok = (df[c1].notna() & df[c2].notna()).to_numpy()
            X = np.column_stack([np.ones(ok.sum()), np.log(df.loc[ok, c1]), np.log(df.loc[ok, c2])])
            r2 = _trimmed_r2(X, ly[ok])
            if r2 >= 0.98:
                for c in (c1, c2):
                    derived.setdefault(c, f"with {c2 if c == c1 else c1} R2={r2:.3f}")
        for c, why in derived.items():
            i = res.index[res["column"] == c][0]
            res.at[i, "flags"] = " | ".join(x for x in [res.at[i, "flags"], f"DERIVED_FROM_TARGET? ({why})"] if x)

    res = res.sort_values("score", ascending=False)
    res["score"] = res["score"].round(4)
    if "missing_indicator_score" in res:
        res["missing_indicator_score"] = res["missing_indicator_score"].round(4)
    res.to_csv(a.out, index=False)
    print(f"rows={len(df)} task={a.task} folds={'GroupKFold' if a.groups else 'KFold'}; baseline score={base}")
    print(res.head(15).to_string(index=False))
    flagged = res[res["flags"] != ""]
    print(f"\n{len(flagged)} flagged. Each needs a record-timing verdict in the leak table. written: {a.out}")


if __name__ == "__main__":
    main()
