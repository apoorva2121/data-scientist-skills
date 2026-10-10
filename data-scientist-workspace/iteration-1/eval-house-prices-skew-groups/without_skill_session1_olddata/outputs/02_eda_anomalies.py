"""
02_eda_anomalies.py - Second EDA pass: outliers / impossible values, repeat-sale structure,
future-dated sales, and what 'groups' mean for validation.
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"
df = pd.read_csv(DATA)
df["sold_date"] = pd.to_datetime(df["sold_date"])
TODAY = pd.Timestamp("2026-09-27")

print("=== sqft distribution tails ===")
print(df["sqft"].quantile([0, .001, .005, .01, .02, .05, .5, .95, .98, .99, .995, .999, 1]))
print("sqft < 300:", (df["sqft"] < 300).sum(), " sqft > 10000:", (df["sqft"] > 10000).sum())
print(df.loc[(df["sqft"] < 300) | (df["sqft"] > 10000), ["sqft", "bedrooms", "bathrooms", "tax_assessed_value", "price", "price_per_sqft"]].sort_values("sqft"))
print("\nsqft per bedroom, for rows with sqft in sane range:")
sane = df[(df.sqft >= 300) & (df.sqft <= 10000)]
print(sane.groupby("bedrooms")["sqft"].describe())

print("\n=== year_built tails ===")
print(df["year_built"].quantile([0, .001, .005, .01, .5, .99, .995, .999, 1]))
print("year_built > sold year:", (df["year_built"] > df["sold_date"].dt.year).sum())
print("year_built > 2026:", (df["year_built"] > 2026).sum())
print("year_built < 1850:", (df["year_built"] < 1850).sum())
print(df.loc[(df["year_built"] > 2026) | (df["year_built"] < 1850), ["year_built", "sold_date", "sqft", "price"]].sort_values("year_built"))

print("\n=== distance tails ===")
print(df["distance_to_center_km"].quantile([0, .5, .95, .99, .995, .999, 1]))
print(df.loc[df["distance_to_center_km"] > 30, ["distance_to_center_km", "neighborhood", "price"]].sort_values("distance_to_center_km"))
print("\nmedian distance by neighborhood (to see if distance is neighborhood-level):")
print(df.groupby("neighborhood")["distance_to_center_km"].agg(["median", "std", "min", "max"]).sort_values("median").head(12))

print("\n=== lot_size tails ===")
print(df["lot_size_sqft"].quantile([0, .01, .5, .99, 1]))
print("Is lot_size missingness related to price? median price missing vs present:")
print(df.groupby(df["lot_size_sqft"].isna())["price"].median())

print("\n=== tax_assessed_value tails ===")
print(df["tax_assessed_value"].quantile([0, .001, .01, .5, .99, .999, 1]))

print("\n=== future-dated sales (after today 2026-09-27) ===")
fut = df[df["sold_date"] > TODAY]
print(len(fut))
print(fut[["listing_id", "sold_date", "price"]].sort_values("sold_date"))
print("\nMonthly counts 2025-2026:")
print(df[df.sold_date >= "2025-01-01"].groupby(df.sold_date.dt.to_period("M")).size())

print("\n=== repeat sales structure ===")
rep = df[df["property_id"].duplicated(keep=False)].sort_values(["property_id", "sold_date"])
print("n properties with >1 sale:", rep["property_id"].nunique())
print("sales per property:", df["property_id"].value_counts().value_counts())
print("\nSample repeat-sale pairs:")
print(rep.head(12)[["property_id", "listing_id", "sold_date", "sqft", "bedrooms", "bathrooms", "year_built", "neighborhood", "condition", "tax_assessed_value", "price"]])

# Are static attributes identical across repeat sales?
static_cols = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft", "neighborhood", "distance_to_center_km"]
same = rep.groupby("property_id")[static_cols].nunique(dropna=False).eq(1).mean()
print("\nFraction of repeat-sale properties with identical static attrs:")
print(same)
# price similarity within property
g = rep.groupby("property_id")["price"]
within = (g.max() - g.min()) / g.mean()
print("\nWithin-property relative price range (max-min)/mean:")
print(within.describe())
# Does tax assessed change?
gt = rep.groupby("property_id")["tax_assessed_value"]
print("\nWithin-property tax_assessed nunique dist:", gt.nunique().value_counts().to_dict())
# Does the listing/property suffix mismatch identify the repeat sales?
mism = df[df["listing_id"].str[1:] != df["property_id"].str[1:]]
print("\nRows where listing_id suffix != property_id suffix:", len(mism))
print("...of which are repeat-sale rows:", mism["property_id"].duplicated(keep=False).sum() + mism["property_id"].isin(rep["property_id"]).sum() - mism["property_id"].duplicated(keep=False).sum())

print("\n=== neighborhood x time (is there drift) ===")
print(df.assign(y=df.sold_date.dt.year).pivot_table(index="neighborhood", columns="y", values="price", aggfunc="median").head(8))

print("\n=== condition ordering check: median price/tax ratio by condition ===")
print((df["price"] / df["tax_assessed_value"]).groupby(df["condition"]).median().sort_values())

print("\n=== log price residual after tax ratio: how much is left to explain? ===")
lp = np.log(df["price"]); lt = np.log(df["tax_assessed_value"])
b = np.polyfit(lt, lp, 1)
res = lp - np.polyval(b, lt)
print("slope/intercept:", b, " resid std (log):", res.std(), " -> ~%.1f%% typical error" % (100 * (np.exp(res.std()) - 1)))
