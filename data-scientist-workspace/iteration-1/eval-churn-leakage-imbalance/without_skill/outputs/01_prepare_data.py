"""
01_prepare_data.py

Loads the raw churn.csv, documents data-quality issues found during EDA,
cleans the data, and writes a cleaned CSV to this outputs folder for the
modeling scripts to consume.

Key EDA findings that drive the cleaning decisions below (see report.md
for full narrative):

1. LEAKAGE: `account_status_code` is almost a perfect encoding of the
   target (code 'C' == cancelled/churned in 527/531 churned rows and only
   39/5036 non-churned rows). It is effectively a post-outcome field and
   is dropped entirely.
2. LEAKAGE: `churn_reason` is populated only for rows where churned == 1
   (its missing rate, 91.35%, exactly equals the non-churn rate). This is
   an outcome field (why the customer churned) and is dropped entirely.
3. DUPLICATE ROWS: 150 rows are exact full-row duplicates of another row
   (same customer_id, identical values). These are dropped (keep first),
   otherwise the same customer could land in both train and test splits.
4. SENTINEL / INVALID VALUES:
   - `age`: 29 rows have age == 0 and 33 rows have age == 250. Both are
     out-of-range placeholder values (max plausible age otherwise ~ 90s)
     and are recoded to missing.
   - `total_charges`: 367 rows have the exact value -999.0, a sentinel
     for missing, on top of ~4% genuine NaNs. Recoded to missing.
5. INCONSISTENT CATEGORY LABELS: `contract_type` has 12 distinct raw
   spellings of 3 real categories (e.g. 'Two year ', 'TWO YEAR',
   'two year', ' One year' ...). Normalized to
   {month-to-month, one year, two year}.
6. REDUNDANT / SUSPECT FEATURE: `last_invoice_amount` is 97% correlated
   with `monthly_charges` (near-duplicate signal) AND its missingness
   rate is almost 2x higher for churned customers (30.5%) than for
   retained customers (16.4%) -- i.e. whether the field is populated at
   all is itself entangled with the outcome, most likely because churned
   customers stop generating invoices. To avoid smuggling outcome
   information in via a missingness pattern, this column is dropped.
7. `customer_id` (identifier) and `signup_date` are dropped: customer_id
   carries no signal, and signup_date was checked against
   signup_date + tenure_months and does not resolve to a consistent
   "snapshot date" across rows (implied snapshot dates span 2023-2031),
   indicating tenure_months and signup_date were generated independently
   -- signup_date adds noise, not a reliable cohort/seasonality signal.

Final modeling features:
  numeric:     tenure_months, monthly_charges, total_charges, age,
               support_calls_90d, avg_daily_usage_mb
  categorical: contract_type, payment_method, region
  target:      churned
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv")
OUT_DIR = Path(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-churn-leakage-imbalance\without_skill\outputs")

LEAKAGE_COLS = ["account_status_code", "churn_reason"]
DROP_COLS = ["customer_id", "signup_date", "last_invoice_amount"] + LEAKAGE_COLS

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "age", "support_calls_90d", "avg_daily_usage_mb",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "region"]
TARGET = "churned"


def normalize_contract_type(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"month-to-month": "month-to-month",
                  "one year": "one year",
                  "two year": "two year"})
    )


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    n_after_dedup = len(df)
    assert df["customer_id"].is_unique, "customer_id should be unique after dropping exact duplicates"

    # sentinel -> missing
    df["age"] = df["age"].replace({0: np.nan, 250: np.nan})
    df["total_charges"] = df["total_charges"].replace({-999.0: np.nan, -999: np.nan})

    # normalize category text
    df["contract_type"] = normalize_contract_type(df["contract_type"])
    df["payment_method"] = df["payment_method"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    print(f"Rows before dedup: {n_before}, after dedup: {n_after_dedup} "
          f"({n_before - n_after_dedup} exact duplicate rows removed)")
    print(f"Final cleaned shape: {df.shape}")
    print(f"Churn rate after cleaning: {df[TARGET].mean():.4f}")
    print("Missing rate per column after cleaning:")
    print(df.isna().mean().sort_values(ascending=False))

    return df


def main():
    df_raw = pd.read_csv(RAW_PATH)
    df_clean = clean(df_raw)

    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df_clean = df_clean[keep_cols]

    out_path = OUT_DIR / "churn_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nSaved cleaned dataset to: {out_path}")


if __name__ == "__main__":
    main()
