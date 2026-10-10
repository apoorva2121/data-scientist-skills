"""
01_eda.py
Exploratory data analysis of store_sales.csv.
Read-only: only inspects the data, does not modify it.
"""
import pandas as pd
import numpy as np

SRC = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"

df = pd.read_csv(SRC, parse_dates=["week_start"])

print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== head ===")
print(df.head(10))

print("\n=== stores ===")
print(sorted(df["store_id"].unique()))
print("n_stores:", df["store_id"].nunique())

print("\n=== date range ===")
print(df["week_start"].min(), df["week_start"].max())
print("n unique weeks:", df["week_start"].nunique())

print("\n=== rows per store ===")
print(df.groupby("store_id").size().describe())
print(df.groupby("store_id").size().sort_values())

print("\n=== missing values ===")
print(df.isna().sum())

print("\n=== duplicate (store_id, week_start) ===")
dups = df.duplicated(subset=["store_id", "week_start"]).sum()
print("dup rows:", dups)

print("\n=== sales stats overall ===")
print(df["sales"].describe())

print("\n=== sales stats per store ===")
print(df.groupby("store_id")["sales"].describe())

print("\n=== negative / zero sales ===")
print((df["sales"] <= 0).sum())
print(df[df["sales"] <= 0])

print("\n=== promo / holiday_week value counts ===")
print(df["promo"].value_counts(dropna=False))
print(df["holiday_week"].value_counts(dropna=False))

print("\n=== avg_temp_c stats ===")
print(df["avg_temp_c"].describe())
print("missing temp:", df["avg_temp_c"].isna().sum())

print("\n=== inventory_end stats ===")
print(df["inventory_end"].describe())
print("negative inventory_end:", (df["inventory_end"] < 0).sum())

# check week continuity per store (gaps)
print("\n=== week continuity check ===")
full_weeks = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="W-MON")
print("expected weeks if continuous weekly Mon-start:", len(full_weeks))
print(full_weeks.min(), full_weeks.max())

for sid, g in df.groupby("store_id"):
    wk = g["week_start"].sort_values()
    diffs = wk.diff().dropna().unique()
    if len(diffs) > 1 or (len(diffs) == 1 and diffs[0] != np.timedelta64(7, 'D')):
        print(sid, "irregular diffs:", diffs, "n_rows:", len(g))

print("\n=== outlier check (z-score per store on sales) ===")
def zscore_outliers(g):
    z = (g["sales"] - g["sales"].mean()) / g["sales"].std()
    return (z.abs() > 4).sum()

outlier_counts = df.groupby("store_id").apply(zscore_outliers)
print(outlier_counts[outlier_counts > 0])

print("\n=== sample rows sorted by sales ascending/descending ===")
print(df.sort_values("sales").head(10))
print(df.sort_values("sales").tail(10))

print("\nDone.")
