"""Stage 5 profile: where the extreme values are, and whether they look like errors, tails or signal.

Usage:
    python outlier_profile.py TRAIN.csv --target TARGET --task binary|multiclass|regression
        [--exclude id_cols] [--time DATE_COL] [--out outlier_profile.csv]

Training rows only. Per numeric column (target included, marked):
  skew                 sample skewness (heavy tails suggest a transform rather than removal)
  iqr15 / iqr3         rows beyond Q1/Q3 -/+ 1.5 or 3 IQR
  mad35                rows with |robust z| > 3.5, robust z = 0.6745 (x - median) / MAD
  flagged_target_rate  target rate (classification) or mean target (regression) among iqr3 | mad35 rows
  overall_target_rate  same over all rows
  lift                 flagged / overall; far from 1 means the extremes carry target information
  max_period_share     largest share of the flagged rows falling in one quarter of the time range
  hint                 entry/unit errors? / heavy tail / signal / isolated extremes (a hint, not a verdict)

Then a multivariate pass: IsolationForest on median-imputed, robust-scaled numeric columns; the top 1%
of anomaly scores is reported with its target rate and the column that deviates most for each row.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    ap.add_argument("--exclude", default="")
    ap.add_argument("--time", default="")
    ap.add_argument("--out", default="outlier_profile.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    exclude = {c.strip() for c in a.exclude.split(",") if c.strip()}
    if a.task == "regression":
        y = df[a.target].astype(float)
    else:
        codes = pd.factorize(df[a.target], sort=True)[0]
        minority = np.argmin(np.bincount(codes))
        y = pd.Series((codes == minority).astype(float), index=df.index)  # rate of the rarest class
    t = pd.to_datetime(df[a.time]) if a.time else None
    num = [c for c in df.columns if c not in exclude and c != a.time and pd.api.types.is_numeric_dtype(df[c])
           and df[c].nunique() > 10 and (c != a.target or a.task == "regression")]

    rows = []
    for c in num:
        x = df[c]
        ok = x.notna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        f15 = ok & ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr))
        f3 = ok & ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr))
        fm = ok & (robust_z(x).abs() > 3.5)
        flag = f3 | fm
        overall = y[ok].mean()
        flagged = y[flag].mean() if flag.any() else np.nan
        lift = flagged / overall if flag.any() and overall else np.nan
        period_share = np.nan
        if t is not None and flag.sum() >= 5:
            quarter = pd.cut(t, 4, labels=False)
            period_share = quarter[flag].value_counts(normalize=True).max()
        skew = float(x.skew())
        n_flag = int(flag.sum())
        p99 = x.quantile(0.99)
        far = int((x > 10 * p99).sum()) if p99 > 0 else 0  # a cluster an order of magnitude beyond p99
        if n_flag == 0:
            hint = "no extremes"
        elif far >= 3 and c != a.target:
            hint = f"{far} rows > 10x p99: likely entry/unit errors (x10, x100, x1000?); verify, then null as errors"
        elif c == a.target:
            hint = "TARGET extremes: robust loss or log target; never drop from test"
        elif pd.notna(lift) and (lift > 1.5 or lift < 0.67) and n_flag >= 10:
            hint = "signal: extremes carry target information; keep (maybe add a flag feature)"
        elif abs(skew) > 1 and f15.sum() > 0.02 * ok.sum():
            hint = "heavy tail: transform (log / Yeo-Johnson) or winsorize for linear models"
        elif n_flag <= max(3, 0.002 * ok.sum()):
            hint = "isolated extremes: check each against domain rules (error?)"
        else:
            hint = "extremes: screen winsorize vs none"
        if pd.notna(period_share) and period_share > 0.6:
            hint += " | clustered in one period (collection issue?)"
        rows.append({"column": c + (" [target]" if c == a.target else ""), "skew": round(skew, 2),
                     "iqr15": int(f15.sum()), "iqr3": int(f3.sum()), "mad35": int(fm.sum()),
                     "min": x.min(), "p01": round(x.quantile(0.01), 3), "p99": round(x.quantile(0.99), 3), "max": x.max(),
                     "flagged_target_rate": round(flagged, 4) if pd.notna(flagged) else np.nan,
                     "overall_target_rate": round(overall, 4), "lift": round(lift, 2) if pd.notna(lift) else np.nan,
                     "max_period_share": round(period_share, 2) if pd.notna(period_share) else np.nan, "hint": hint})
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    label = "mean target" if a.task == "regression" else "rarest-class rate"
    print(f"rows={len(df)}; target summary uses {label}")
    print(res.drop(columns=["min", "max"]).to_string(index=False))

    feats = [c for c in num if c != a.target]
    if len(feats) >= 2:
        Z = df[feats].apply(lambda s: s.fillna(s.median()))
        Z = Z.apply(robust_z)
        iso = IsolationForest(n_estimators=300, random_state=0).fit(Z)
        score = -iso.score_samples(Z)
        top = score >= np.quantile(score, 0.99)
        driver = Z[top].abs().idxmax(axis=1).value_counts()
        print(f"\nIsolationForest top 1% ({int(top.sum())} rows): {label} {y[top].mean():.4f} vs {y.mean():.4f} overall")
        print("most-deviating column among them:", ", ".join(f"{k}={v}" for k, v in driver.items()))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
