"""
Stage 3 - Leakage hunt + relationship EDA, train rows only (ds-data-audit Part B).

Runs leak_screen.py on split/train.csv (already train-only), then reproduces the
relationship EDA (target rate by decile/level, redundancy) used to write leak_table.csv
and eda_notes.md.
"""
import subprocess, sys
import pandas as pd

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
LEAK_SCREEN = r"C:/dev/important repo/Data Scientist Skill Set/ds-data-audit/scripts/leak_screen.py"

# 1. leak screen (train-only; excludes ID and date columns)
subprocess.run([sys.executable, LEAK_SCREEN, OUT + "/split/train.csv",
                 "--target", "churned", "--task", "binary",
                 "--exclude", "customer_id,signup_date",
                 "--out", OUT + "/leak_screen.csv"], check=True)

# 2. relationship EDA, train only
train = pd.read_csv(OUT + "/split/train.csv")
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
             "avg_daily_usage_mb", "age", "last_invoice_amount"]

print("=== target rate by decile ===")
for c in num_feats:
    q = pd.qcut(train[c], 10, duplicates="drop")
    print(c)
    print(train.groupby(q, observed=True)["churned"].agg(["mean", "count"]))
    print()

print("=== target rate by category ===")
for c in ["contract_type", "payment_method", "region"]:
    print(c)
    print(train.groupby(c)["churned"].agg(["mean", "count"]).sort_values("mean", ascending=False))
    print()

print("=== redundancy: spearman among numerics ===")
print(train[num_feats].corr(method="spearman").round(2))

# 3. stability spot-check: first half vs second half of train (by row order, proxy for
#    "does the relationship hold on an independent subsample")
half = len(train) // 2
for c in ["contract_type"]:
    print(f"\nstability check on {c}:")
    print("first half:\n", train.iloc[:half].groupby(c)["churned"].mean())
    print("second half:\n", train.iloc[half:].groupby(c)["churned"].mean())

# See leak_table.csv and eda_notes.md for the written verdicts and findings.
