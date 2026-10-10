"""
Exploratory data analysis for house_prices.csv.
Run and its stdout redirected to eda_output.txt for the record.
"""
import numpy as np
import pandas as pd

RAW_PATH = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

df = pd.read_csv(RAW_PATH)

print("=== shape / dtypes ===")
print(df.shape)
print(df.dtypes)

print("\n=== missing values ===")
print(df.isna().sum())

print("\n=== duplicate property_id (repeat sales) ===")
print("rows:", len(df), "unique property_id:", df.property_id.nunique(), "unique listing_id:", df.listing_id.nunique())
print(df.property_id.value_counts().value_counts())
dup_ids = df.property_id.value_counts()
dup_ids = dup_ids[dup_ids == 2].index[:3]
print(df[df.property_id.isin(dup_ids)].sort_values(["property_id", "sold_date"]).to_string())

print("\n=== garage raw values ===")
print(df.garage.value_counts(dropna=False))

print("\n=== condition values ===")
print(df.condition.value_counts(dropna=False))

print("\n=== neighborhood cardinality ===")
print(df.neighborhood.nunique(), "levels")
print(df.neighborhood.value_counts())

print("\n=== price_per_sqft vs price/sqft (leakage check) ===")
calc = df.price / df.sqft
print((calc - df.price_per_sqft).abs().describe())

print("\n=== target: price ===")
print(df.price.describe())
print("skew:", df.price.skew())

print("\n=== sold_date range ===")
print(df.sold_date.min(), df.sold_date.max())

print("\n=== numeric summary ===")
print(df[["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
          "distance_to_center_km", "tax_assessed_value", "price"]].describe().to_string())

print("\n=== sqft outliers (sentinel check) ===")
print(df.sqft.sort_values(ascending=False).head(15).to_string())
print("rows with sqft==100000:", (df.sqft == 100000).sum())
bad_sqft = df[df.sqft == 100000][["sqft", "bedrooms", "bathrooms", "price", "tax_assessed_value"]]
print(bad_sqft.to_string())

print("\n=== year_built outliers (sentinel check) ===")
print("unique low values:", sorted(df.year_built.unique())[:5])
print("unique high values:", sorted(df.year_built.unique())[-5:])
print("rows year_built==1800:", (df.year_built == 1800).sum())
print("rows year_built==2090:", (df.year_built == 2090).sum())

print("\n=== correlation with price (raw numeric) ===")
num = df[["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
          "distance_to_center_km", "tax_assessed_value", "price"]]
print(num.corr()["price"].sort_values(ascending=False))

print("\n=== tax_assessed_value / price ratio (is it a leaky duplicate of price?) ===")
print((df.tax_assessed_value / df.price).describe())
