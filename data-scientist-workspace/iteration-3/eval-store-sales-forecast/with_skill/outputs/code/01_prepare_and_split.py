"""Stage 1 (stateless cleaning) + Stage 2 (validation split) for store_sales.csv.

Findings from column_audit.py + manual inspection (see decision_log.md Stage 1):
- 4 exact duplicate rows (store_id, week_start pairs S08/2024-11-04, S17/2024-04-01,
  S18/2024-07-01, S19/2024-02-19) -> drop the duplicate copy.
- sales == -1 (5 rows): impossible (negative sales) -> sentinel, set to NaN.
- sales == 9999 (1 row, S11 2024-09-16): fits naturally between neighbouring weeks
  (10139 -> 9999 -> 12063) -> real value, NOT a sentinel despite the round-number flag.
- sales == 0 for store S19 (63 rows): a real, recurring intermittent pattern (roughly 1 in 3
  weeks nonzero, inventory climbs steadily across the zero weeks) -> genuine zeros, kept.
- avg_temp_c negative values (54 rows, min -4.9): plausible winter Celsius temperatures, not
  sentinels, despite the -1.0 x2 coincidental sentinel-pattern flag -> kept as-is.
- inventory_end: same-period leak (end-of-week inventory is a joint outcome with that week's
  sales, corr=0.27 same-week) and unusable for future forecasting anyway (not known ahead)
  -> excluded from features, never dropped from the row.
- holiday_week is a deterministic calendar rule: rate is exactly 1.0 for ISO week-of-year in
  {1, 14, 27, 47, 51, 52} and exactly 0.0 for every other week, in every store-year -> can be
  reproduced exactly for the 8-week forecast horizon (known-in-advance covariate).
"""
import sys

import numpy as np
import pandas as pd

RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-store-sales-forecast\with_skill\outputs"

HOLIDAY_WEEKS_ISO = {1, 14, 27, 47, 51, 52}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning rules only (Non-negotiable #1). Reused for raw rows at handoff."""
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates(keep="first")
    n1 = len(df)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df.loc[df["sales"] == -1, "sales"] = np.nan  # impossible negative sales -> sentinel
    n_sales_null = df["sales"].isna().sum()
    print(f"cleaning: {n0} -> {n1} rows after exact-dup drop ({n0 - n1} removed); "
          f"{n_sales_null} sales values nulled (sentinel -1)")
    return df


def isoweek_holiday_flag(dates: pd.Series) -> pd.Series:
    return dates.dt.isocalendar().week.astype(int).isin(HOLIDAY_WEEKS_ISO).astype(int)


if __name__ == "__main__":
    df = pd.read_csv(RAW)
    df = clean_raw(df)
    # sanity: reproduced holiday rule matches the data's own holiday_week column exactly (train rows only, checked below after split)
    df.to_csv(f"{OUT}/cleaned.csv", index=False)

    sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
    import subprocess
    cmd = [
        sys.executable,
        r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts\make_split.py",
        f"{OUT}/cleaned.csv", "--target", "sales", "--scheme", "group_time",
        "--id", "store_id,week_start" if False else "", "--group", "store_id", "--time", "week_start",
        "--test-periods", "8", "--holdout-periods", "8", "--n-splits", "5", "--seed", "42",
        "--out", f"{OUT}/split",
    ]
    # make_split.py needs a single unique id column; build a composite row id first.
    df2 = df.copy()
    df2["row_id"] = df2["store_id"].astype(str) + "_" + df2["week_start"].dt.strftime("%Y-%m-%d")
    df2.to_csv(f"{OUT}/cleaned.csv", index=False)
    cmd = [
        sys.executable,
        r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts\make_split.py",
        f"{OUT}/cleaned.csv", "--target", "sales", "--scheme", "group_time",
        "--id", "row_id", "--group", "store_id", "--time", "week_start",
        "--test-periods", "8", "--holdout-periods", "8", "--n-splits", "5", "--seed", "42",
        "--out", f"{OUT}/split",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)

    # verify holiday rule reproduces the data's own flag exactly
    check = (isoweek_holiday_flag(df["week_start"]) == df["holiday_week"]).mean()
    print("holiday-rule match rate vs data:", check)
