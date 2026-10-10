"""Step 2: targeted checks on missingness-as-leakage, duplicates, and sentinel values."""
import pandas as pd
import numpy as np

CSV = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
df = pd.read_csv(CSV)

# Duplicates: are they exact copies with identical labels?
dups = df[df["customer_id"].duplicated(keep=False)].sort_values("customer_id")
print("dup rows:", len(dups), "unique ids among them:", dups["customer_id"].nunique())
print("all dup pairs identical on every column:", dups.groupby("customer_id").nunique().max().max() == 1)
print("churn rate in dup pairs:", dups.drop_duplicates("customer_id")["churned"].mean())

# Missingness vs churn
for c in ["last_invoice_amount", "total_charges"]:
    m = df[c].isna()
    print(f"\n{c}: missing={m.sum()}  churn rate | missing = {df.loc[m,'churned'].mean():.3f}  "
          f"| present = {df.loc[~m,'churned'].mean():.3f}   share missing among churners = {m[df.churned==1].mean():.3f} "
          f"vs non-churners = {m[df.churned==0].mean():.3f}")

# -999 sentinel in total_charges
s = df["total_charges"] == -999
print(f"\ntotal_charges == -999: {s.sum()}  churn rate | sentinel = {df.loc[s,'churned'].mean():.3f}")
print("negative total_charges values other than -999:", df.loc[(df.total_charges < 0) & ~s, "total_charges"].unique()[:10])

# Age anomalies
print("\nage <18:", (df.age < 18).sum(), " age==0:", (df.age == 0).sum(), " age>100:", (df.age > 100).sum())
print("churn rate | age==0:", df.loc[df.age == 0, "churned"].mean(), " | age>100:", df.loc[df.age > 100, "churned"].mean())
print("age values >100:", sorted(df.loc[df.age > 100, "age"].unique()))

# Usage outliers
print("\nusage >100:", (df.avg_daily_usage_mb > 100).sum(), " churn rate:", df.loc[df.avg_daily_usage_mb > 100, "churned"].mean())
print("usage >100 values:", sorted(df.loc[df.avg_daily_usage_mb > 100, "avg_daily_usage_mb"].unique())[:20])

# account_status_code: S vs A churn rates, and combined with leak removal, how much does status carry?
print("\nchurn rate by status:", df.groupby("account_status_code")["churned"].mean().round(4).to_dict())

# last_invoice vs monthly charges for churners vs not (is invoice systematically different?)
r = df["last_invoice_amount"] / df["monthly_charges"]
print("\ninvoice/monthly ratio, churn=1 vs 0:", r[df.churned == 1].mean().round(4), r[df.churned == 0].mean().round(4))

# Interactions worth knowing: support calls x churn, tenure buckets x churn
print("\nchurn by support_calls_90d:\n", df.groupby("support_calls_90d")["churned"].agg(["mean", "size"]))
print("\nchurn by tenure bucket:\n", df.groupby(pd.cut(df.tenure_months, [0, 6, 12, 24, 36, 48, 60, 72]))["churned"].agg(["mean", "size"]))
print("\nchurn by monthly_charges quintile:\n", df.groupby(pd.qcut(df.monthly_charges, 5))["churned"].agg(["mean", "size"]))
print("\nchurn by usage quintile:\n", df.groupby(pd.qcut(df.avg_daily_usage_mb, 5))["churned"].agg(["mean", "size"]))
