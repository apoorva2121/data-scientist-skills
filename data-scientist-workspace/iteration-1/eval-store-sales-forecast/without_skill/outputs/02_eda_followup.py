"""
02_eda_followup.py
Follow-up investigation on specific data issues found in 01_eda.py:
- duplicate (store_id, week_start) rows
- S03 missing weeks
- S19 near-zero / zero sales pattern
- negative sales sentinel values
"""
import pandas as pd
import numpy as np

SRC = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
df = pd.read_csv(SRC, parse_dates=["week_start"])

print("=== duplicate (store_id, week_start) rows, full content ===")
dup_mask = df.duplicated(subset=["store_id", "week_start"], keep=False)
print(df[dup_mask].sort_values(["store_id", "week_start"]))

print("\n=== S03 full timeline (check missing weeks) ===")
s03 = df[df.store_id == "S03"].sort_values("week_start")
full_range = pd.date_range(s03.week_start.min(), s03.week_start.max(), freq="7D")
missing = set(full_range) - set(s03.week_start)
print("missing weeks for S03:", sorted(missing))
print("n rows S03:", len(s03))

print("\n=== S19 timeline sample ===")
s19 = df[df.store_id == "S19"].sort_values("week_start")
print(s19[["week_start","sales","promo","holiday_week","inventory_end"]].to_string())
print("\nS19 zero-sales count:", (s19["sales"]==0).sum(), "/", len(s19))
print("S19 inventory_end trend (first 5, last 5):")
print(s19[["week_start","inventory_end"]].head())
print(s19[["week_start","inventory_end"]].tail())

print("\n=== negative sales rows (sentinel -1) ===")
neg = df[df["sales"] < 0]
print(neg)
print("count:", len(neg))
print("stores affected:", neg["store_id"].unique())

print("\n=== zero sales rows by store ===")
zero = df[df["sales"] == 0]
print(zero["store_id"].value_counts())

print("\n=== check if inventory_end ~ inventory_start - sales + restock (sanity, informal) ===")
# just look at correlation of sales with inventory drawdown per store as sanity check
for sid in ["S00", "S19", "S12"]:
    g = df[df.store_id == sid].sort_values("week_start").reset_index(drop=True)
    g["inv_prev"] = g["inventory_end"].shift(1)
    g["implied_receipt_plus_prevsales"] = g["inv_prev"] - g["inventory_end"] + g["sales"]
    print(sid, "corr(sales, inventory_end):", g["sales"].corr(g["inventory_end"]))

print("\nDone.")
