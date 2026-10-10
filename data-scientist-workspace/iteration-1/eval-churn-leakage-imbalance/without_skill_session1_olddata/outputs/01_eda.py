"""Step 1: exploratory data analysis of churn.csv."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

CSV = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
df = pd.read_csv(CSV)

print("shape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nmissing:\n", df.isna().sum())
print("\nduplicated customer_id:", df["customer_id"].duplicated().sum())
print("\nduplicated full rows:", df.duplicated().sum())
print("\ntarget rate:", df["churned"].mean(), df["churned"].value_counts().to_dict())

print("\ndescribe numeric:\n", df.describe().T)

for c in ["contract_type", "payment_method", "region", "account_status_code", "churn_reason"]:
    print(f"\n--- {c} ---")
    vc = df[c].value_counts(dropna=False)
    print(vc.head(30))
    print("unique raw values (repr):", [repr(v) for v in df[c].dropna().unique()[:30]])

# leakage checks
print("\n=== churn_reason vs churned ===")
print(pd.crosstab(df["churn_reason"].fillna("<NA>"), df["churned"]))
print("\n=== account_status_code vs churned ===")
print(pd.crosstab(df["account_status_code"], df["churned"]))

# signup_date
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
print("\nsignup_date range:", df["signup_date"].min(), df["signup_date"].max(), "NaT:", df["signup_date"].isna().sum())
print("churn rate by signup year-month (last 12):")
print(df.groupby(df["signup_date"].dt.to_period("M"))["churned"].agg(["mean", "size"]).tail(15))

# tenure vs signup date consistency: tenure_months should ~= months from signup to some snapshot date
ref = df["signup_date"] + pd.to_timedelta(df["tenure_months"] * 30.44, unit="D")
print("\nimplied snapshot date (signup + tenure): min/median/max:", ref.min(), ref.median(), ref.max())

# total_charges vs tenure*monthly
df["implied_total"] = df["tenure_months"] * df["monthly_charges"]
df["tc_ratio"] = df["total_charges"] / df["implied_total"].replace(0, np.nan)
print("\ntotal_charges / (tenure*monthly) ratio quantiles:\n", df["tc_ratio"].describe())
print("rows with tenure==0:", (df["tenure_months"] == 0).sum())

# last_invoice vs monthly charges
df["inv_ratio"] = df["last_invoice_amount"] / df["monthly_charges"]
print("\nlast_invoice/monthly ratio by churn:\n", df.groupby("churned")["inv_ratio"].describe())

# univariate signal
num = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age", "last_invoice_amount"]
print("\nmean by churn:\n", df.groupby("churned")[num].mean().T)
for c in ["contract_type", "payment_method", "region"]:
    print(f"\nchurn rate by {c}:\n", df.groupby(c)["churned"].agg(["mean", "size"]).sort_values("mean"))

# odd values
print("\nnegative values:", {c: int((df[c] < 0).sum()) for c in num})
print("age quantiles:", df["age"].quantile([0, .01, .5, .99, 1]).to_dict())
print("support_calls quantiles:", df["support_calls_90d"].quantile([0, .5, .9, .99, 1]).to_dict())
print("usage quantiles:", df["avg_daily_usage_mb"].quantile([0, .01, .5, .99, 1]).to_dict())
