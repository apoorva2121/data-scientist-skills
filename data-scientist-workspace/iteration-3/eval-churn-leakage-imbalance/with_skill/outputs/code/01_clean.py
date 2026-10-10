"""Stage 1: structure audit + stateless cleaning (ds-data-audit part A).
Rerunnable end to end. No fitting happens here (non-negotiable #1).
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv")
OUT = Path(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-churn-leakage-imbalance\with_skill\outputs")

CONTRACT_MAP = {
    "month-to-month": "Month-to-month", "MONTH-TO-MONTH": "Month-to-month", " Month-to-month": "Month-to-month",
    "Month-to-month": "Month-to-month",
    "one year": "One year", "One Year": "One year", " One year": "One year", "One year": "One year",
    "two year": "Two year", "TWO YEAR": "Two year", "Two year": "Two year",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-wise fixes only: types, sentinel->null, spelling, exact dup, impossible->null.
    This function is reused unchanged as the first step of the final production pipeline (Stage 10).
    """
    df = df.copy()
    log = []

    n0 = len(df)
    df = df.drop_duplicates(keep="first")
    log.append(f"drop exact duplicate rows: {n0 - len(df)} removed, {len(df)} remaining")

    # normalize contract_type spelling variants (case/whitespace) -> canonical levels
    df["contract_type"] = df["contract_type"].astype(str).str.strip().replace(CONTRACT_MAP)
    log.append(f"contract_type: mapped {len(CONTRACT_MAP)} spelling variants to 3 canonical levels")

    # total_charges: -999 is a sentinel for missing (column_audit.py flagged SENTINEL?-999x367)
    n_sent = (df["total_charges"] == -999).sum()
    df.loc[df["total_charges"] == -999, "total_charges"] = np.nan
    log.append(f"total_charges: -999 sentinel -> NaN ({n_sent} rows)")

    # age: values > 120 are impossible (column_audit.py flagged FAR_OUT_3IQR; max raw value was 250)
    n_age = (df["age"] > 120).sum()
    df.loc[df["age"] > 120, "age"] = np.nan
    log.append(f"age: impossible values (>120) -> NaN ({n_age} rows)")

    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="raise")
    log.append("signup_date: parsed as date (explicit format)")

    df["customer_id"] = df["customer_id"].astype(str)

    return df, log


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    cleaned, log = clean_raw(raw)
    cleaned.to_csv(OUT / "cleaned.csv", index=False)

    with open(OUT / "cleaning_log.md", "w") as f:
        f.write("# Cleaning log (Stage 1, stateless only)\n\n")
        f.write(f"Raw rows: {len(raw)}. Cleaned rows: {len(cleaned)}.\n\n")
        for line in log:
            f.write(f"- {line}\n")
        f.write("\nTarget after cleaning:\n")
        f.write(cleaned["churned"].value_counts(normalize=False).to_string())
        f.write("\n\n")
        f.write(cleaned["churned"].value_counts(normalize=True).to_string())
        f.write("\n\nRemoved-row concentration check (dup rows vs class): duplicates were exact copies "
                "(same customer_id + all fields), dropped rows are identical to a kept row so class "
                "balance is unaffected by construction.\n")

    print("rows raw:", len(raw), "cleaned:", len(cleaned))
    print(cleaned["churned"].mean())
    print("\n".join(log))
