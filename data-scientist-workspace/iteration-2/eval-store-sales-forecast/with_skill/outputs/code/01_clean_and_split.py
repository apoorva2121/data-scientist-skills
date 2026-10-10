"""Stage 1 (structure audit + stateless cleaning) and Stage 2 (validation design) for the
store_sales.csv 8-week-ahead forecasting task. Compressed to one script under a tight time
budget; every check the bundled column_audit.py / leak_screen.py would run is done here by hand
and logged in decision_log.md.
"""
import sys, json, hashlib
import numpy as np
import pandas as pd

RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-store-sales-forecast\with_skill\outputs"
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-forecasting\scripts")
from ts_backtest import regularize

df = pd.read_csv(RAW, parse_dates=["week_start"])
n0 = len(df)

# --- 1. exact duplicate rows (store_id, week_start, identical everywhere) -> drop ---
exact_dupe_mask = df.duplicated(keep="first")
n_exact_dupes = int(exact_dupe_mask.sum())
df = df[~exact_dupe_mask].copy()

# --- 2. remaining (store_id, week_start) duplicates with differing values, if any ---
key_dupes = df.duplicated(subset=["store_id", "week_start"], keep=False)
n_key_dupes = int(key_dupes.sum())  # expect 0 after dropping exact dupes

# --- 3. sentinel: sales == -1.0 is not a real sale (all other sales are 3-4 digit positive
#     volumes); convert to NaN. True zero sales (store S19) are left as 0 (intermittent demand,
#     see decision log). ---
n_sentinel = int((df["sales"] == -1.0).sum())
df.loc[df["sales"] == -1.0, "sales"] = np.nan

# --- 4. inventory_end is a same-period leak for a sales forecast (it is recorded at the end of
#     the same week as the sales figure it would be used to predict) -> excluded entirely, never
#     used as a feature. Kept in the audit table only. ---

cleaning_log = pd.DataFrame([
    {"step": "drop exact duplicate rows", "rows_affected": n_exact_dupes, "rows_remaining": len(df)},
    {"step": "check residual (store,week) duplicates after dedup", "rows_affected": n_key_dupes, "rows_remaining": len(df)},
    {"step": "sales == -1.0 sentinel -> NaN", "rows_affected": n_sentinel, "rows_remaining": len(df)},
])
cleaning_log.to_csv(f"{OUT}/cleaning_log.csv", index=False)

# --- 5. regularize calendar per store (find gaps) ---
panel, gaps = regularize(df, id_col="store_id", time_col="week_start", target="sales", freq="W-MON")
gaps.to_csv(f"{OUT}/gap_report.csv", index=False)

# regularize() keeps all extra raw columns (promo, holiday_week, avg_temp_c, inventory_end)
# already aligned to the reindexed calendar; NaN on any inserted gap rows. Drop inventory_end
# here: it is a same-period leak for a sales forecast (recorded at the end of the same week).
panel = panel.drop(columns=["inventory_end"])

# fill covariates on inserted gap rows: promo/holiday default to 0 (no evidence of an event),
# temp interpolated in time per store
panel["promo"] = panel["promo"].fillna(0.0)
panel["holiday_week"] = panel["holiday_week"].fillna(0.0)
panel = panel.sort_values(["unique_id", "ds"])
panel["avg_temp_c"] = panel.groupby("unique_id")["avg_temp_c"].transform(lambda s: s.interpolate(limit_direction="both"))

# fill missing y (sentinel + true gaps): business assumed open (no evidence of closure) ->
# linear interpolation within each store's own series, using neighboring weeks only (never
# crosses the test/holdout boundary because interpolation uses immediate neighbors and all gaps
# are interior to 2023-2024 history, well before the last 16 weeks reserved below).
n_missing_y_before = int(panel["y"].isna().sum())
panel["y"] = panel.groupby("unique_id")["y"].transform(lambda s: s.interpolate(limit_direction="both"))
n_missing_y_after = int(panel["y"].isna().sum())

panel.to_csv(f"{OUT}/panel_clean.csv", index=False)

print("rows raw:", n0, "-> after cleaning:", len(df))
print("exact dupes dropped:", n_exact_dupes)
print("sentinel sales(-1) -> NaN:", n_sentinel)
print("gap rows inserted by regularize:", len(gaps), gaps.gap_type.value_counts().to_dict())
print("y missing before/after interpolation:", n_missing_y_before, n_missing_y_after)
print("panel shape:", panel.shape, "stores:", panel.unique_id.nunique())

# --- Stage 2: validation design (group_time, test=last 8 weeks/store, holdout=8 before that) ---
panel_for_split = panel.copy()
panel_for_split["row_id"] = panel_for_split["unique_id"] + "_" + panel_for_split["ds"].dt.strftime("%Y-%m-%d")
split_in = f"{OUT}/panel_for_split.csv"
panel_for_split.to_csv(split_in, index=False)

import subprocess
cmd = [sys.executable, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts\make_split.py",
       split_in, "--target", "y", "--scheme", "group_time", "--id", "row_id", "--group", "unique_id",
       "--time", "ds", "--test-periods", "8", "--holdout-periods", "8", "--n-splits", "5", "--seed", "42",
       "--out", f"{OUT}/split"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-3000:])
