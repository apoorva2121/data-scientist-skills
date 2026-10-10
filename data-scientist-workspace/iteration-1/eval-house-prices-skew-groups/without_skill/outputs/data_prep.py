"""
Shared data loading / cleaning / feature engineering for the house price
pricing-tool project.

Key data quality findings from EDA (see 01_eda.py for the full printout):

- 4320 rows, 4000 unique `property_id`. 320 properties appear twice (sold
  twice in the window). Rows sharing a property_id are near-duplicates in
  every feature except sold_date/price, so any row-level random split leaks
  the same property into both train and test and inflates validation
  scores. -> all splits in this project are GroupKFold / GroupShuffleSplit
  on property_id.
- `price_per_sqft` is (up to rounding) price / sqft, i.e. computed directly
  from the target. It is dropped -- keeping it would be target leakage.
- `garage` is a messy categorical: {Yes, No, Y, YES, yes, N, no, NaN}. It is
  normalized to {Yes, No, Unknown}, with a `garage_missing` flag.
- `sqft` has 13 rows fixed at exactly 100000, wildly inconsistent with the
  otherwise normal sale price for those bedroom/bathroom counts. Treated as
  a sentinel/error code -> set to NaN + `sqft_outlier_flag`.
- `year_built` has 13 rows at exactly 2090 (future, impossible) and 9 rows
  at exactly 1800, both disconnected from the real 1900-2023 range seen in
  the rest of the data. Treated as sentinel/error codes -> set to NaN +
  `year_built_outlier_flag`.
- `lot_size_sqft` is missing for ~20% of rows, with no obvious relationship
  to price (missing-value mean price ~= observed-value mean price) ->
  treated as MAR, imputed with a `lot_size_missing` flag kept as a feature.
- All numeric imputation is done with sklearn SimpleImputer *inside* the
  model pipeline (fit on train folds only) to avoid leaking calibration/test
  information into training.
"""

import numpy as np
import pandas as pd

RAW_PATH = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"

NUMERIC_FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size_sqft",
    "distance_to_center_km",
    "tax_assessed_value",
    "house_age",
    "sale_year",
    "sale_month",
    "sqft_outlier_flag",
    "year_built_outlier_flag",
    "lot_size_missing",
    "garage_missing",
]
CATEGORICAL_FEATURES = ["neighborhood", "condition", "garage_clean"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"
GROUP_COL = "property_id"


def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- temporal features ---
    df["sold_date"] = pd.to_datetime(df["sold_date"])
    df["sale_year"] = df["sold_date"].dt.year
    df["sale_month"] = df["sold_date"].dt.month

    # --- garage: normalize free-text-ish categories ---
    garage_map = {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}
    garage_clean = df["garage"].astype(str).str.strip().str.lower().map(garage_map)
    df["garage_missing"] = garage_clean.isna().astype(int)
    df["garage_clean"] = garage_clean.fillna("Unknown")

    # --- sqft sentinel (100000 placeholder) ---
    df["sqft_outlier_flag"] = (df["sqft"] >= 50000).astype(int)
    df.loc[df["sqft_outlier_flag"] == 1, "sqft"] = np.nan

    # --- year_built sentinel (1800 / 2090 placeholders) ---
    df["year_built_outlier_flag"] = (
        (df["year_built"] <= 1850) | (df["year_built"] > 2025)
    ).astype(int)
    df.loc[df["year_built_outlier_flag"] == 1, "year_built"] = np.nan

    # --- engineered feature: house age at sale ---
    df["house_age"] = df["sale_year"] - df["year_built"]

    # --- lot size missingness flag (imputation happens in-pipeline) ---
    df["lot_size_missing"] = df["lot_size_sqft"].isna().astype(int)

    # --- drop target-leaking derived column ---
    df = df.drop(columns=["price_per_sqft"])

    return df


def get_X_y_groups(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups


if __name__ == "__main__":
    d = load_and_clean()
    print(d.shape)
    print(d[ALL_FEATURES + [TARGET]].isna().sum())
