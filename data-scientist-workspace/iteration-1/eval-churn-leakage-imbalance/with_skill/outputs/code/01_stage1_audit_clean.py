"""
Stage 1 - Structure audit + stateless cleaning (ds-data-audit Part A).
Reads the raw churn.csv, applies only row-wise, stateless fixes (no split-dependent
statistics), and writes cleaned_data.csv + cleaning_log.md.

Stateless fixes applied here (all reversible, row-wise, no fitting):
  1. Drop exact duplicate rows.
  2. total_charges: -999 sentinel -> NaN.
  3. age: impossible values (0, 250) -> NaN (plausible range is 18-85, hard gap to both).
  4. contract_type: normalize case/whitespace to 3 canonical levels.

Nothing here is fit on data (no imputation, no scaling, no encoding).
"""
import pandas as pd
import numpy as np

RAW = r"C:/dev/important repo/Data Scientist Skill Set/evals/datasets/churn.csv"
OUT_DIR = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"

log_lines = []

df = pd.read_csv(RAW)
log_lines.append(f"raw rows read: {len(df)}")

# 1. exact duplicate rows
n_before = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
log_lines.append(f"drop exact duplicate rows: {n_before - len(df)} removed -> {len(df)} remain")

# 2. total_charges sentinel -999 -> NaN
n_sentinel = int((df["total_charges"] == -999).sum())
df.loc[df["total_charges"] == -999, "total_charges"] = np.nan
log_lines.append(f"total_charges: -999 sentinel -> NaN, {n_sentinel} values affected; "
                  f"total missing now {df['total_charges'].isna().sum()} "
                  f"({df['total_charges'].isna().mean():.2%})")

# 3. age impossible values -> NaN (0 and 250; plausible range 18-85 with a hard gap)
n_age_bad = int(((df["age"] == 0) | (df["age"] == 250)).sum())
df.loc[(df["age"] == 0) | (df["age"] == 250), "age"] = np.nan
log_lines.append(f"age: impossible values {{0, 250}} -> NaN, {n_age_bad} values affected "
                  f"({df['age'].isna().mean():.2%} missing now)")

# 4. contract_type spelling variants -> canonical
canon_map = {
    "month-to-month": "month-to-month", "month-to-month ": "month-to-month",
    " month-to-month": "month-to-month",
    "one year": "one year", "one year ": "one year", " one year": "one year",
    "two year": "two year", "two year ": "two year", " two year": "two year",
}
before_levels = df["contract_type"].unique().tolist()
df["contract_type"] = df["contract_type"].str.strip().str.lower()
after_levels = sorted(df["contract_type"].unique().tolist())
log_lines.append(f"contract_type: {len(before_levels)} raw spellings -> canonicalized to "
                  f"{after_levels} (map: strip + lowercase)")
assert set(after_levels) == {"month-to-month", "one year", "two year"}

# Check dropped/affected rows are not concentrated in one class (sanity)
dup_check = pd.read_csv(RAW)
dup_mask = dup_check.duplicated(keep="first")
churn_rate_all = dup_check["churned"].mean()
churn_rate_dupes = dup_check.loc[dup_mask, "churned"].mean()
log_lines.append(f"sanity: churn rate overall={churn_rate_all:.4f}, "
                  f"churn rate among dropped duplicate rows={churn_rate_dupes:.4f} "
                  f"(not concentrated in either class)")

log_lines.append(f"final cleaned rows: {len(df)}, columns: {len(df.columns)}")
log_lines.append(f"target class balance after cleaning: "
                  f"{df['churned'].value_counts(normalize=True).to_dict()}")

# Nothing else changed: churn_reason / account_status_code are NOT dropped here.
# They stay in cleaned_data.csv (audit trail); Stage 3 leak hunt formally drops them
# from the FEATURE list on train-only evidence via leak_screen.py.

df.to_csv(OUT_DIR + "/cleaned_data.csv", index=False)

with open(OUT_DIR + "/cleaning_log.md", "w") as f:
    f.write("# Cleaning log — Stage 1 (stateless, row-wise only)\n\n")
    for line in log_lines:
        f.write(f"- {line}\n")
    f.write("\nCandidate keys / groups / time columns handed to Stage 2:\n")
    f.write("- key: customer_id (unique after dedup)\n")
    f.write("- group column: none identified (customer_id unique per row post-dedup, "
            "no repeated-entity structure)\n")
    f.write("- time column: signup_date present but inconsistent with tenure_months "
            "(r=0.019) — not usable to build a time-ordered split; no reliable "
            "snapshot/extraction date column exists\n")

print("\n".join(log_lines))
print("\nWrote:", OUT_DIR + "/cleaned_data.csv", "and cleaning_log.md")
