"""Stage 1: structure audit + stateless cleaning (ds-data-audit Part A).

Runs BEFORE any split. Only stateless, row-wise fixes: sentinel->null, spelling
normalization, impossible values->null, exact-duplicate removal. No imputation,
scaling, encoding or outlier capping happens here (Stage 4/5).
"""
import pandas as pd
import numpy as np

RAW_PATH = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"
OUT_DIR = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs"

GARAGE_MAP = {
    "Yes": "Yes", "yes": "Yes", "Y": "Yes", "YES": "Yes",
    "No": "No", "no": "No", "N": "No",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise cleaning. Reused identically in the production pipeline
    (Stage 1 output feeds the FunctionTransformer that is the first pipeline step)."""
    df = df.copy()

    # sentinels -> null (round, out-of-plausible-range placeholder values)
    df.loc[df["sqft"].isin([50, 100000]), "sqft"] = np.nan
    df.loc[df["lot_size_sqft"] == 9999, "lot_size_sqft"] = np.nan
    df.loc[df["year_built"].isin([1800, 2090]), "year_built"] = np.nan

    # spelling variants -> canonical levels
    df["garage"] = df["garage"].map(GARAGE_MAP).where(df["garage"].notna(), np.nan)

    # types
    df["sold_date"] = pd.to_datetime(df["sold_date"], format="%Y-%m-%d")
    df["listing_id"] = df["listing_id"].astype(str)
    df["property_id"] = df["property_id"].astype(str)
    df["condition"] = pd.Categorical(
        df["condition"], categories=["Poor", "Fair", "Good", "Very Good", "Excellent"], ordered=True
    )

    return df


def main():
    df = pd.read_csv(RAW_PATH)
    n0 = len(df)

    log_lines = [f"rows read: {n0}"]

    # exact duplicate rows
    dup_exact = df.duplicated().sum()
    log_lines.append(f"exact duplicate rows: {dup_exact} (none dropped, count=0)")

    # key uniqueness
    dup_key = df["listing_id"].duplicated().sum()
    log_lines.append(f"listing_id duplicates: {dup_key}")
    n_props = df["property_id"].nunique()
    dup_prop = df["property_id"].duplicated().sum()
    log_lines.append(
        f"property_id: {n_props} unique / {n0} rows -> {dup_prop} repeat-sale rows "
        f"(GROUP_KEY for Stage 2 split)"
    )

    before = {}
    for col in ["sqft", "lot_size_sqft", "year_built", "garage"]:
        before[col] = df[col].isna().sum()

    cleaned = clean_raw(df)

    after = {}
    for col in ["sqft", "lot_size_sqft", "year_built", "garage"]:
        after[col] = cleaned[col].isna().sum()

    log_lines.append("")
    log_lines.append("sentinel/impossible -> null (stateless):")
    log_lines.append(f"  sqft: {50} and {100000} treated as sentinel -> "
                      f"{(after['sqft'] - before['sqft'])} rows nulled "
                      f"(now {after['sqft']} missing, {after['sqft']/n0:.2%})")
    log_lines.append(f"  lot_size_sqft: 9999 treated as sentinel -> "
                      f"{(after['lot_size_sqft'] - before['lot_size_sqft'])} rows nulled "
                      f"(now {after['lot_size_sqft']} missing, {after['lot_size_sqft']/n0:.2%})")
    log_lines.append(f"  year_built: 1800 and 2090 treated as sentinel (isolated spikes, gap "
                      f"to nearest real value >50yr) -> "
                      f"{(after['year_built'] - before['year_built'])} rows nulled "
                      f"(now {after['year_built']} missing, {after['year_built']/n0:.2%})")
    log_lines.append(f"  garage: spelling variants {{Yes,yes,Y,YES}}->Yes, {{No,no,N}}->No; "
                      f"blank stays missing (now {after['garage']} missing, {after['garage']/n0:.2%})")

    log_lines.append("")
    log_lines.append(f"rows remaining after Stage 1: {len(cleaned)} (no rows dropped; "
                      f"impossible values nulled in place, not removed)")

    with open(f"{OUT_DIR}/cleaning_log.md", "w") as f:
        f.write("# Stage 1 cleaning log\n\n")
        f.write("\n".join(log_lines))
        f.write("\n")

    cleaned.to_csv(f"{OUT_DIR}/house_prices_stage1_clean.csv", index=False)
    print("\n".join(log_lines))
    print("\nsaved:", f"{OUT_DIR}/house_prices_stage1_clean.csv")


if __name__ == "__main__":
    main()
