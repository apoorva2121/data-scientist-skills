"""Stage 10: handoff. Wrap the Stage 1 stateless cleaning rules as the first step of the saved
pipeline, so raw production rows get exactly the cleaning the training data got. Score a handful
of raw rows end to end; confirm identical output when the same row is scored twice.
"""
import sys, json
import numpy as np, pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-churn-leakage-imbalance\with_skill\outputs"
RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"

CONTRACT_MAP = {
    "month-to-month": "Month-to-month", "MONTH-TO-MONTH": "Month-to-month", " Month-to-month": "Month-to-month",
    "Month-to-month": "Month-to-month",
    "one year": "One year", "One Year": "One year", " One year": "One year", "One year": "One year",
    "two year": "Two year", "TWO YEAR": "Two year", "Two year": "Two year",
}

def clean_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, stateless production cleaning: sentinel->null, spelling->canonical,
    impossible->null. No dedup here (dedup is a training-set operation on a whole table, not a
    per-row rule); no fitting. Safe to apply to a single incoming row or a batch."""
    df = df.copy()
    df["contract_type"] = df["contract_type"].astype(str).str.strip().replace(CONTRACT_MAP)
    df["total_charges"] = df["total_charges"].astype(float)
    df.loc[df["total_charges"] == -999, "total_charges"] = np.nan
    df["age"] = df["age"].astype(float)
    df.loc[df["age"] > 120, "age"] = np.nan
    return df

# joblib.load here reads only the file this same run wrote in 05_final_eval.py (this project's
# own output folder), not an untrusted or externally supplied artifact, so pickle deserialization
# risk does not apply.
trained = joblib.load(f"{OUT}/final_pipeline.joblib")  # pre + model, fitted on cleaned train+holdout
prod_pipe = Pipeline([("clean", FunctionTransformer(clean_raw_features)), *trained.steps])
joblib.dump(prod_pipe, f"{OUT}/final_pipeline_production.joblib")
print("saved final_pipeline_production.joblib (clean_raw_features -> pre -> model)")

# score a handful of RAW, uncleaned rows end to end (includes a -999 total_charges row and an
# age=250 impossible row if present) - confirm output + latency, and repeat-row stability
raw = pd.read_csv(RAW)
sample = pd.concat([
    raw[raw["total_charges"] == -999].head(2),
    raw[raw["age"] > 120].head(1),
    raw.sample(5, random_state=0),
]).reset_index(drop=True)

import time
t0 = time.time()
scores = prod_pipe.predict_proba(sample)[:, 1]
elapsed = time.time() - t0
print(f"scored {len(sample)} raw rows in {elapsed*1000:.1f} ms ({elapsed/len(sample)*1000:.2f} ms/row)")
for cid, sc in zip(sample["customer_id"], scores):
    print(f"  {cid}: churn_risk={sc:.4f}")

# feed one row twice, confirm identical score
row_twice = pd.concat([sample.iloc[[0]], sample.iloc[[0]]], ignore_index=True)
s2 = prod_pipe.predict_proba(row_twice)[:, 1]
assert np.isclose(s2[0], s2[1]), "same row scored twice gave different scores!"
print(f"repeat-row check passed: {s2[0]:.6f} == {s2[1]:.6f}")

with open(f"{OUT}/handoff_check.json", "w") as f:
    json.dump({"n_scored": len(sample), "ms_per_row": elapsed/len(sample)*1000,
               "repeat_row_identical": bool(np.isclose(s2[0], s2[1])),
               "sample_scores": {str(c): float(s) for c, s in zip(sample["customer_id"], scores)}}, f, indent=2)
print("written handoff_check.json")
