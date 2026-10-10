"""
03_feature_checks.py
Check whether holiday_week / promo / avg_temp_c carry usable signal and
whether holiday_week is deterministic by calendar week (so it could be
treated as "known in advance" for a future forecast).
"""
import pandas as pd
import numpy as np

SRC = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
df = pd.read_csv(SRC, parse_dates=["week_start"])
df = df.drop_duplicates(subset=["store_id", "week_start"]).copy()
df["woy"] = df["week_start"].dt.isocalendar().week.astype(int)
df["year"] = df["week_start"].dt.year

print("=== holiday_week by iso week-of-year, across years ===")
piv = df.groupby(["woy"])["holiday_week"].agg(["mean", "sum", "count"])
print(piv[piv["mean"] > 0])

print("\n=== is holiday_week identical across years for each woy? ===")
cross = df.groupby(["woy", "year"])["holiday_week"].mean().unstack()
mismatch = cross[(cross.nunique(axis=1) > 1)]
print("weeks where holiday flag differs by year (should be ~0 rows if deterministic):")
print(mismatch.dropna(how="all"))

print("\n=== promo rate by store (is it store-random or seasonal?) ===")
print(df.groupby("store_id")["promo"].mean())

print("\n=== correlation checks (overall, pooled, naive) ===")
print("corr(sales, promo):", df["sales"].corr(df["promo"]))
print("corr(sales, holiday_week):", df["sales"].corr(df["holiday_week"]))
print("corr(sales, avg_temp_c):", df["sales"].corr(df["avg_temp_c"]))

print("\n=== within-store correlation of sales with temp (detrended-ish, per store) ===")
for sid, g in df.groupby("store_id"):
    g = g.sort_values("week_start")
    c_temp = g["sales"].corr(g["avg_temp_c"])
    c_promo = g["sales"].corr(g["promo"])
    c_hol = g["sales"].corr(g["holiday_week"])
    print(f"{sid}: temp={c_temp:.2f} promo={c_promo:.2f} holiday={c_hol:.2f}")

print("\n=== average promo uplift within store (mean sales promo=1 vs promo=0) ===")
for sid, g in df.groupby("store_id"):
    m1 = g.loc[g.promo == 1, "sales"].mean()
    m0 = g.loc[g.promo == 0, "sales"].mean()
    print(f"{sid}: promo_on={m1:.0f} promo_off={m0:.0f} uplift%={100*(m1-m0)/m0:.1f}")

print("\nDone.")
