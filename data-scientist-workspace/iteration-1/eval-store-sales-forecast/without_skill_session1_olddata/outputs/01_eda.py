"""
01_eda.py - Data profiling for store_sales.csv

Purpose: understand the structure, quality and signal in the data before
choosing a forecasting approach. Prints findings to stdout; nothing is modelled here.
"""
import pandas as pd
import numpy as np

DATA = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\store_sales.csv"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv(DATA)
print("=== SHAPE / DTYPES ===")
print(df.shape)
print(df.dtypes)
print()

print("=== HEAD ===")
print(df.head())
print()

df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
print("=== UNPARSEABLE DATES ===", df["week_start"].isna().sum())
print()

print("=== MISSING VALUES ===")
print(df.isna().sum())
print()

print("=== STORES ===")
print("n stores:", df["store_id"].nunique(), sorted(df["store_id"].unique()))
print()

print("=== DATE RANGE OVERALL ===")
print(df["week_start"].min(), "->", df["week_start"].max())
print("distinct weeks:", df["week_start"].nunique())
print()

print("=== PER-STORE ROWS / DATE RANGE ===")
g = df.groupby("store_id")["week_start"].agg(["count", "min", "max", "nunique"])
print(g)
print()

print("=== DUPLICATE (store, week) ROWS ===")
dups = df[df.duplicated(["store_id", "week_start"], keep=False)].sort_values(["store_id", "week_start"])
print(len(dups))
if len(dups):
    print(dups.head(20))
print()

print("=== DAY-OF-WEEK OF week_start (should be uniform, e.g. all Monday) ===")
print(df["week_start"].dt.day_name().value_counts())
print()

print("=== WEEKLY CALENDAR GAPS PER STORE (expected 7-day spacing) ===")
for sid, sub in df.sort_values("week_start").groupby("store_id"):
    d = sub["week_start"].drop_duplicates().diff().dt.days.dropna()
    bad = d[d != 7]
    if len(bad):
        print(sid, "irregular spacing counts:", bad.value_counts().to_dict())
full_idx = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="7D")
print("expected weeks if complete:", len(full_idx))
missing_by_store = {}
for sid, sub in df.groupby("store_id"):
    have = set(sub["week_start"])
    miss = [d for d in full_idx if d not in have]
    if miss:
        missing_by_store[sid] = miss
print("stores with missing weeks:", {k: len(v) for k, v in missing_by_store.items()})
for k, v in list(missing_by_store.items())[:20]:
    print(k, [str(x.date()) for x in v])
print()

print("=== NUMERIC SUMMARY ===")
print(df[["promo", "holiday_week", "avg_temp_c", "sales", "inventory_end"]].describe().T)
print()

print("=== NON-POSITIVE / EXTREME SALES ===")
print("sales <= 0:", (df["sales"] <= 0).sum())
print("sales NaN:", df["sales"].isna().sum())
print("negative inventory:", (df["inventory_end"] < 0).sum())
# robust per-store outlier detection: |x - median| / MAD
def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-9)
df["sales_rz"] = df.groupby("store_id")["sales"].transform(robust_z)
out = df[df["sales_rz"].abs() > 4].sort_values("sales_rz")
print("rows with |robust z| > 4:", len(out))
print(out[["store_id", "week_start", "promo", "holiday_week", "avg_temp_c", "sales", "inventory_end", "sales_rz"]].to_string())
print()

print("=== PROMO / HOLIDAY FREQUENCY ===")
print("promo value counts:", df["promo"].value_counts().to_dict())
print("holiday value counts:", df["holiday_week"].value_counts().to_dict())
print("promo rate by store:")
print(df.groupby("store_id")["promo"].mean().round(3).to_dict())
print("holiday weeks (distinct dates):")
print(sorted(df.loc[df["holiday_week"] == 1, "week_start"].dt.date.unique()))
print("is holiday flag consistent across stores for a given week?")
hw = df.groupby("week_start")["holiday_week"].nunique()
print("weeks where holiday flag differs across stores:", (hw > 1).sum())
pw = df.groupby("week_start")["promo"].nunique()
print("weeks where promo flag differs across stores:", (pw > 1).sum(), "of", len(pw))
print()

print("=== EFFECT SIZES (mean sales ratio) ===")
for col in ["promo", "holiday_week"]:
    m = df.groupby(["store_id", col])["sales"].mean().unstack()
    ratio = (m[1] / m[0]).describe()
    print(col, "uplift ratio across stores:\n", ratio.round(3))
print()

print("=== STORE LEVELS ===")
lvl = df.groupby("store_id")["sales"].agg(["mean", "median", "std", "min", "max"])
lvl["cv"] = lvl["std"] / lvl["mean"]
print(lvl.round(1))
print()

print("=== TREND: first-half vs second-half mean per store ===")
mid = df["week_start"].median()
tr = df.assign(half=np.where(df["week_start"] <= mid, "h1", "h2")).groupby(["store_id", "half"])["sales"].mean().unstack()
tr["ratio_h2_h1"] = tr["h2"] / tr["h1"]
print(tr.round(3))
print()

print("=== SEASONALITY: mean sales by month (all stores, normalised by store mean) ===")
df["norm_sales"] = df["sales"] / df.groupby("store_id")["sales"].transform("mean")
print(df.groupby(df["week_start"].dt.month)["norm_sales"].mean().round(3))
print("by ISO week-of-year (first 10 / last 10):")
wk = df.groupby(df["week_start"].dt.isocalendar().week)["norm_sales"].mean().round(3)
print(wk.head(10).to_dict())
print(wk.tail(10).to_dict())
print()

print("=== TEMPERATURE ===")
print("temp by month:", df.groupby(df["week_start"].dt.month)["avg_temp_c"].mean().round(1).to_dict())
print("corr(norm_sales, temp) overall:", round(df[["norm_sales", "avg_temp_c"]].corr().iloc[0, 1], 3))
print("is temp identical across stores in a given week?", (df.groupby("week_start")["avg_temp_c"].nunique() == 1).mean())
print()

print("=== SALES <= 0 BREAKDOWN ===")
z = df[df["sales"] <= 0]
print(z.groupby("store_id")["sales"].agg(["count", "min", "max"]))
print("distinct non-positive values:", sorted(z["sales"].unique()))
print()

print("=== S19 FULL PROFILE (suspicious store) ===")
s19 = df[df["store_id"] == "S19"].sort_values("week_start")
print(s19[["week_start", "promo", "holiday_week", "sales", "inventory_end"]].to_string())
print()

print("=== AUTOCORRELATION of normalised sales (pooled, per-store demeaned) ===")
dd = df.drop_duplicates(["store_id", "week_start"])
for lag in [1, 2, 4, 8, 13, 26, 52]:
    ac = []
    for sid, sub in dd.sort_values("week_start").groupby("store_id"):
        if sid == "S19":
            continue
        s = sub.set_index("week_start")["sales"].asfreq("7D")
        s = (s - s.mean())
        ac.append(s.autocorr(lag))
    print(f"lag {lag:>2}: mean autocorr = {np.nanmean(ac):.3f}")
print()

print("=== INVENTORY vs SALES ===")
print("corr(sales, inventory_end) within store (mean):",
      round(df.groupby("store_id").apply(lambda s: s["sales"].corr(s["inventory_end"]), include_groups=False).mean(), 3))
# does inventory_end + sales look like a constant / does inventory lag predict sales?
df = df.sort_values(["store_id", "week_start"])
df["inv_lag1"] = df.groupby("store_id")["inventory_end"].shift(1)
print("corr(sales, inventory_end lag1) within store (mean):",
      round(df.groupby("store_id").apply(lambda s: s["sales"].corr(s["inv_lag1"]), include_groups=False).mean(), 3))
print("mean(sales + inventory_end) per store (is it roughly constant?):")
print((df["sales"] + df["inventory_end"]).groupby(df["store_id"]).agg(["mean", "std"]).round(0).head())
print()

print("=== LAST 10 WEEKS OF ONE STORE ===")
print(df[df["store_id"] == "S00"].tail(10)[["week_start", "promo", "holiday_week", "avg_temp_c", "sales", "inventory_end"]])
