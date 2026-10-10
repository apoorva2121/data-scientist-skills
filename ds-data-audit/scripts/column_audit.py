"""Stage 1 column audit: one row per column with the facts needed to decide stateless cleaning.

Usage:
    python column_audit.py DATA.csv --target TARGET [--key COL1,COL2] [--out column_audit.csv] [--sep ,]

Reads everything as raw strings first so sentinels and spelling variants are not hidden by
pandas' own parsing, then infers types. Prints a short summary and writes the table.
It does not modify the data: cleaning decisions are made by the analyst and logged.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

NUMERIC_SENTINELS = {-1, -9, -99, -999, -9999, -99999, 999, 9999, 99999, 999999}
STRING_SENTINELS = {"", "na", "n/a", "nan", "null", "none", "nil", "?", "-", "--", "unknown", "missing", "#n/a", "undefined"}
DATE_SENTINELS = ("1900-01-01", "1970-01-01", "2099-12-31", "9999-12-31")


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


def _density_spikes(vals: pd.Series, min_ratio: float = 50.0, min_share: float = 0.005,
                    window: int = 25, low_card_share: float = 0.05,
                    low_card_ratio: float = 8.0) -> list[str]:
    """Flag values carrying far more rows than their neighbours in sorted order.

    Catches what the sentinel patterns cannot: a censoring floor, a clipping ceiling or an
    imputed default that happens to be an ordinary-looking number. A floor at 30,000 holding 9%
    of the rows, whose next distinct value holds two, matches no -999 pattern and is invisible
    to every other check here - but it is the most important structural fact about that column.

    Two criteria, either of which fires:
      * continuous columns - count >= `min_ratio` x the median neighbour count AND share >= `min_share`
      * discrete columns (tens of distinct values, every neighbour already holds ~1-2%) - share >=
        `low_card_share` AND count >= `low_card_ratio` x the median neighbour count. Without this a
        20% floor in an integer age column is only ~15x its neighbours and would be missed.
    Results are ranked by share so a ceiling is never dropped behind heaping values at the low end.
    Columns with under 10 distinct values are skipped: there every value is common and the ratio
    carries no information. Columns that are unique per row are skipped: nothing can spike.
    """
    if vals.nunique() == len(vals):
        return []
    vc = vals.value_counts().sort_index()
    if len(vc) < 10:
        return []
    counts = vc.to_numpy(dtype=float)
    values = vc.index.to_numpy()
    total = counts.sum()
    # rolling median of the neighbours, excluding self, vectorised via a padded window
    k = len(counts)
    out = []
    for i in np.flatnonzero(counts >= min(low_card_ratio, min_ratio) * 1.0):  # cheap prefilter
        lo, hi = max(0, i - window), min(k, i + window + 1)
        neigh = np.concatenate([counts[lo:i], counts[i + 1:hi]])
        if neigh.size == 0:
            continue
        med = float(np.median(neigh))
        cnt = counts[i]
        share = cnt / total
        where = "min" if i == 0 else "max" if i == k - 1 else "mid"
        # A placeholder is ISOLATED: the values immediately beside it hold far fewer rows. A
        # natural mode of a skewed discrete distribution (a Poisson count, say) is not - its
        # neighbours hold nearly as many rows as it does - yet a windowed median dragged down by
        # the thin tail can make it look 10x "above its neighbours". Require isolation for the
        # discrete and edge criteria; the 50x continuous rule implies it already.
        adjacent = max(counts[i - 1] if i > 0 else 0.0, counts[i + 1] if i < k - 1 else 0.0)
        isolated = adjacent < 0.5 * cnt
        cont = share >= min_share and cnt >= min_ratio * max(med, 1.0)
        disc = isolated and share >= low_card_share and cnt >= low_card_ratio * max(med, 1.0)
        # Censoring and top-coding live at the ends of the range, so a pile-up AT the minimum or
        # maximum is diagnostic at a much lower ratio than one in the middle: a 5% top-code in an
        # integer column is only ~3-4x its neighbours and would otherwise slip through.
        edge = isolated and where != "mid" and share >= 0.03 and cnt >= 3.0 * max(med, 1.0)
        if cont or disc or edge:
            out.append((share, f"DENSITY_SPIKE@{values[i]:g}:{int(cnt)}rows({share:.1%})"
                               f"_vs_neighbour_median_{med:.0f}_at_{where}"))
    out.sort(key=lambda z: -z[0])
    return [s for _, s in out[:3]]


def audit_column(name: str, raw: pd.Series, n_rows: int) -> dict:
    s = raw.astype("string")
    stripped = s.str.strip()
    is_blank = stripped.isna() | (stripped == "")
    non_blank = stripped[~is_blank]

    num = _to_numeric(non_blank) if len(non_blank) else pd.Series(dtype=float)
    numeric_share = float(num.notna().mean()) if len(non_blank) else 0.0
    dt = pd.to_datetime(non_blank, errors="coerce", format="mixed") if 0 < len(non_blank) and numeric_share < 0.5 else None
    date_share = float(dt.notna().mean()) if dt is not None else 0.0

    if numeric_share >= 0.95:
        kind = "numeric"
    elif date_share >= 0.95:
        kind = "datetime"
    else:
        kind = "categorical"

    row = {
        "column": name,
        "inferred_type": kind,
        "n_missing_raw": int(is_blank.sum()),
        "pct_missing_raw": round(100 * is_blank.mean(), 2),
        "n_unique": int(non_blank.nunique()),
        "pct_unique": round(100 * non_blank.nunique() / max(n_rows, 1), 2),
        "top_values": "; ".join(f"{k}={v}" for k, v in non_blank.value_counts().head(3).items()),
    }

    flags = []
    if row["n_unique"] <= 1:
        flags.append("CONSTANT")
    if row["pct_unique"] >= 95 and kind != "numeric":
        flags.append("ID_LIKE")
    elif kind == "categorical" and row["pct_unique"] >= 30:
        counts = non_blank.value_counts()
        if (counts > 1).any() and counts.median() <= 5:
            flags.append(f"GROUP_KEY?:{int((counts > 1).sum())}_values_repeat")
    if kind == "numeric" and row["pct_unique"] >= 95 and num.dropna().is_monotonic_increasing:
        flags.append("ID_LIKE_MONOTONIC")

    if kind == "numeric":
        vals = num.dropna()
        row.update(
            min=vals.min(), p01=vals.quantile(0.01), median=vals.median(), p99=vals.quantile(0.99), max=vals.max(),
            skew=round(float(vals.skew()), 2) if len(vals) > 2 else np.nan,
        )
        hits = {v: int((vals == v).sum()) for v in NUMERIC_SENTINELS if (vals == v).sum() > 0}
        if hits:
            flags.append("SENTINEL?" + ",".join(f"{k}x{v}" for k, v in sorted(hits.items())))
        non_numeric = non_blank[num.isna()]
        if len(non_numeric):
            flags.append("NON_NUMERIC_TOKENS:" + ",".join(non_numeric.unique()[:5]))
        if (vals < 0).any() and (vals >= 0).mean() > 0.97:
            flags.append(f"RARE_NEGATIVES:{int((vals < 0).sum())}")
        zeros = int((vals == 0).sum())
        if zeros and vals.quantile(0.01) > 0:
            # zeros exist but are rarer than 1% and far below the rest: often a disguised missing code
            flags.append(f"ZERO_VALUES?:{zeros}")
        if len(vals) > 20 and vals.quantile(0.99) > 0:
            ratio = vals.max() / vals.quantile(0.99)
            if ratio > 20:
                flags.append(f"EXTREME_MAX:{ratio:.0f}x_p99")
        flags += _density_spikes(vals)
        looks_like_year = "year" in name.lower() or (
            len(vals) > 20 and (vals % 1 == 0).all() and vals.between(1000, 3000).mean() > 0.99
        )
        if looks_like_year:
            this_year = pd.Timestamp.today().year
            bad = int(((vals < 1850) | (vals > this_year + 1)).sum())
            if bad:
                flags.append(f"YEAR_OUT_OF_RANGE:{bad}")
        if len(vals) > 20:
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                far = int(((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum())
                if far:
                    flags.append(f"FAR_OUT_3IQR:{far}")
    elif kind == "datetime":
        row.update(min=str(dt.min())[:10], max=str(dt.max())[:10])
        hits = [d for d in DATE_SENTINELS if (non_blank.str[:10] == d).any()]
        if hits:
            flags.append("DATE_SENTINEL?" + ",".join(hits))
        future = int((dt > pd.Timestamp.today()).sum())
        if future and not hits:
            flags.append(f"FUTURE_DATES:{future}")
    else:
        lowered = non_blank.str.lower()
        sent = sorted(set(lowered.unique()) & STRING_SENTINELS)
        if sent or is_blank.any():
            tokens = sent + (["<blank>"] if is_blank.any() else [])
            flags.append("STRING_SENTINEL?" + ",".join(tokens))
        canon = non_blank.str.strip().str.lower().str.replace(r"[\s_\-]+", " ", regex=True)
        collisions = non_blank.nunique() - canon.nunique()
        if collisions > 0:
            flags.append(f"SPELLING_VARIANTS:{collisions}")
        is_key = any(f.startswith(("ID_LIKE", "GROUP_KEY")) for f in flags)
        if row["n_unique"] > 50 and not is_key:
            flags.append("HIGH_CARDINALITY")
        rare = (non_blank.value_counts() < max(10, 0.001 * n_rows)).sum()
        if rare and not is_key:
            flags.append(f"RARE_LEVELS:{int(rare)}")
    if row["n_missing_raw"] and not any(f.startswith("STRING_SENTINEL") for f in flags):
        flags.insert(0, f"MISSING:{row['pct_missing_raw']}%")
    row["flags"] = " | ".join(flags)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="column_audit.csv")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--key", default="", help="comma-separated columns that should uniquely identify a row")
    a = ap.parse_args()

    raw = pd.read_csv(a.data, dtype=str, keep_default_na=False, sep=a.sep)
    n = len(raw)
    print(f"rows={n} cols={raw.shape[1]}")
    print(f"exact duplicate rows={int(raw.duplicated().sum())}")
    stripped = raw.apply(lambda c: c.str.strip().str.lower())
    print(f"duplicates after trim/lowercase={int(stripped.duplicated().sum())}")
    if a.key:
        key = [k.strip() for k in a.key.split(",")]
        dup_key = raw.duplicated(key, keep=False)
        conflicting = raw[dup_key].drop_duplicates().duplicated(key, keep=False).sum()
        print(f"rows sharing key {key}={int(dup_key.sum())} (of which conflicting, non-identical={int(conflicting)})")

    table = pd.DataFrame([audit_column(c, raw[c], n) for c in raw.columns])
    table.to_csv(a.out, index=False)

    t = raw[a.target].str.strip()
    tn = _to_numeric(t)
    print(f"\ntarget '{a.target}': missing={int((t == '').sum())} unique={t.nunique()}")
    if tn.notna().mean() > 0.95 and t.nunique() > 20:
        print(tn.describe().round(3).to_string())
        print(f"skew={tn.skew():.2f}  (>1 suggests log / Box-Cox target or a robust loss)")
    else:
        vc = t.value_counts()
        print(pd.DataFrame({"count": vc, "share": (vc / vc.sum()).round(4)}).to_string())

    flagged = table[table["flags"] != ""][["column", "inferred_type", "pct_missing_raw", "flags"]]
    print(f"\nflagged columns ({len(flagged)}):")
    print(flagged.to_string(index=False) if len(flagged) else "none")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
