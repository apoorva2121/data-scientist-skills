"""
01_eda.py - Exploratory data analysis for house_prices.csv
Goal: understand target distribution, missingness, duplicates, leakage candidates,
and the structure of categorical/group columns before modelling.
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"
df = pd.read_csv(DATA)

print("=== SHAPE ===")
print(df.shape)
print("\n=== DTYPES ===")
print(df.dtypes)
print("\n=== HEAD ===")
print(df.head(10))

print("\n=== MISSING (count, pct) ===")
miss = df.isna().sum()
print(pd.DataFrame({"n_missing": miss, "pct": (miss / len(df) * 100).round(2)}))

print("\n=== DESCRIBE numeric ===")
print(df.describe().T)

print("\n=== ID uniqueness ===")
print("listing_id unique:", df["listing_id"].nunique(), "/", len(df))
print("property_id unique:", df["property_id"].nunique(), "/", len(df))
print("listing_id == property_id suffix match:",
      (df["listing_id"].str[1:] == df["property_id"].str[1:]).mean())
print("Full-row duplicates:", df.duplicated().sum())
print("Duplicates ignoring listing_id:", df.drop(columns=["listing_id"]).duplicated().sum())
print("Duplicate property_id rows (repeat sales):", df["property_id"].duplicated(keep=False).sum())

print("\n=== CATEGORICALS ===")
for c in ["bedrooms", "bathrooms", "neighborhood", "condition", "garage"]:
    vc = df[c].value_counts(dropna=False)
    print(f"\n--- {c} ({df[c].nunique(dropna=False)} levels) ---")
    print(vc.head(30))

print("\n=== sold_date ===")
d = pd.to_datetime(df["sold_date"], errors="coerce")
print("unparseable:", d.isna().sum())
print("min:", d.min(), "max:", d.max())
print(d.dt.year.value_counts().sort_index())

print("\n=== TARGET price ===")
p = df["price"]
print(p.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]))
print("skew:", p.skew(), " log-skew:", np.log(p[p > 0]).skew())
print("non-positive prices:", (p <= 0).sum())
print("price NaN:", p.isna().sum())

print("\n=== price_per_sqft check (leakage candidate) ===")
recomputed = df["price"] / df["sqft"]
print("abs diff price_per_sqft vs price/sqft (describe):")
print((df["price_per_sqft"] - recomputed).abs().describe())
print("corr(price_per_sqft, price):", df[["price_per_sqft", "price"]].corr().iloc[0, 1])

print("\n=== tax_assessed_value vs price ===")
print("corr:", df[["tax_assessed_value", "price"]].corr().iloc[0, 1])
ratio = df["price"] / df["tax_assessed_value"]
print("price / tax ratio describe:")
print(ratio.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

print("\n=== Numeric sanity ===")
for c in ["sqft", "lot_size_sqft", "year_built", "distance_to_center_km", "tax_assessed_value"]:
    s = df[c]
    print(f"{c}: min={s.min()}, max={s.max()}, n<=0={(s<=0).sum()}, nunique={s.nunique()}")

print("\n=== Correlations with price (numeric) ===")
num = df.select_dtypes(include=np.number)
print(num.corr()["price"].sort_values(ascending=False))

print("\n=== Price by neighborhood ===")
g = df.groupby("neighborhood")["price"].agg(["count", "median", "mean", "std"]).sort_values("median")
print(g)

print("\n=== Price by condition ===")
print(df.groupby("condition")["price"].agg(["count", "median", "mean"]))

print("\n=== Price by garage (raw labels) ===")
print(df.groupby("garage", dropna=False)["price"].agg(["count", "median", "mean"]))

print("\n=== Price by year sold ===")
print(df.assign(year=d.dt.year).groupby("year")["price"].agg(["count", "median", "mean"]))

print("\n=== sqft vs price outliers ===")
print(df.sort_values("price", ascending=False).head(10)[["sqft","bedrooms","neighborhood","tax_assessed_value","price","price_per_sqft"]])
print(df.sort_values("price").head(10)[["sqft","bedrooms","neighborhood","tax_assessed_value","price","price_per_sqft"]])
