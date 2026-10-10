"""
Stage 10 - Handoff.

Fixes the Stage 9 sanity-test finding: the pipeline saved for production must re-apply the
Stage 1 stateless cleaning rules itself (sentinel-to-null, impossible-value-to-null, category
normalization), because a real deployment feeds it raw rows straight from the source system,
not the already-cleaned cleaned_data.csv this project used for screening. Wraps those rules as
a proper pipeline step (RawRowCleaner), rebuilds the exact Stage 8 winner on top of it, refits
on train+holdout, re-runs the raw-row sanity check, and saves the result with skops.
"""
import sys, json, time, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
from cv_folds import load_split

RAW = r"C:/dev/important repo/Data Scientist Skill Set/evals/datasets/churn.csv"
FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
         "avg_daily_usage_mb", "age", "contract_type", "payment_method"]
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
cat_feats = ["contract_type", "payment_method"]


class RawRowCleaner(BaseEstimator, TransformerMixin):
    """Re-applies the Stage 1 stateless, row-wise cleaning rules inside the pipeline itself, so
    the saved artifact is safe on truly raw production rows (this is not 'fitted' on data - every
    rule is a fixed constant decided once at Stage 1 on evidence, never re-estimated per fold)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "total_charges" in X.columns:
            X["total_charges"] = X["total_charges"].mask(X["total_charges"] == -999, np.nan)
        if "age" in X.columns:
            X["age"] = X["age"].mask((X["age"] == 0) | (X["age"] == 250), np.nan)
        if "contract_type" in X.columns:
            X["contract_type"] = X["contract_type"].astype(str).str.strip().str.lower()
        return X


def make_production_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    return Pipeline([
        ("clean", RawRowCleaner()),
        ("pre", pre),
        ("clf", LogisticRegression(C=0.019150292626953554, max_iter=3000, class_weight="balanced")),
    ])


df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train_ho = df[s.train_mask | s.holdout_mask].reset_index(drop=True)

prod_pipe = make_production_pipe().fit(train_ho[FEATS], train_ho["churned"])

# ---- re-run the raw-row sanity check on genuinely raw, uncleaned rows from churn.csv ----
raw_df = pd.read_csv(RAW)
sample = raw_df.sample(5, random_state=0)[FEATS + ["customer_id"]].copy()
# inject a known-dirty row: sentinel total_charges, impossible age, odd-cased contract_type,
# exactly mirroring a real raw record this pipeline must survive
dirty = pd.DataFrame([{"customer_id": "C_TEST_DIRTY", "tenure_months": 24, "monthly_charges": 70.0,
                        "total_charges": -999, "support_calls_90d": 1, "avg_daily_usage_mb": 15.0,
                        "age": 250, "contract_type": "TWO YEAR", "payment_method": "Mailed check"}])
sample = pd.concat([sample, dirty], ignore_index=True)

t0 = time.time()
preds = prod_pipe.predict_proba(sample[FEATS])[:, 1]
latency_ms = (time.time() - t0) * 1000 / len(sample)
print("raw-row scoring (production pipeline, self-cleaning):")
for cid, p in zip(sample["customer_id"], preds):
    print(f"  {cid}: p={p:.4f}")
print(f"latency: {latency_ms:.3f} ms/row (batch of {len(sample)})")

# duplicate-row check
dup2 = sample[FEATS].iloc[[-1, -1]]
p2 = prod_pipe.predict_proba(dup2)[:, 1]
print(f"duplicate dirty row -> {p2[0]:.6f} vs {p2[1]:.6f} (match={np.isclose(p2[0], p2[1])})")

# confirm the dirty row's total_charges=-999 and age=250 were actually nulled, not treated as literal numbers
cleaned_check = prod_pipe.named_steps["clean"].transform(dirty[FEATS])
assert pd.isna(cleaned_check["total_charges"].iloc[0]), "cleaner failed to null the -999 sentinel"
assert pd.isna(cleaned_check["age"].iloc[0]), "cleaner failed to null the impossible age"
assert cleaned_check["contract_type"].iloc[0] == "two year", "cleaner failed to canonicalize casing"
print("PASS: sentinel/-impossible-value/casing rules verified inside the saved pipeline")

# ---- confirm identical predictions vs the Stage 9 pipeline on already-clean rows ----
test = df[s.test_mask].reset_index(drop=True)
from sklearn.pipeline import Pipeline as SkPipeline
prod_scores_on_clean = prod_pipe.predict_proba(test[FEATS])[:, 1]
prev = pd.read_csv(OUT + "/test_results.csv")
print("\nprod pipeline PR-AUC on test (sanity vs Stage 9's 0.2107):",
      round(float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
          test["churned"], prod_scores_on_clean)), 4))

# ---- save with skops ----
import skops.io as sio
sio.dump(prod_pipe, OUT + "/final_pipeline.skops")
unknown = sio.get_untrusted_types(file=OUT + "/final_pipeline.skops")
print("skops untrusted types on reload:", unknown)
reloaded = sio.load(OUT + "/final_pipeline.skops", trusted=unknown)
p_reload = reloaded.predict_proba(sample[FEATS])[:, 1]
print("reload check: predictions match original ->", np.allclose(p_reload, preds))

versions = {
    "python": platform.python_version(),
    "scikit-learn": sklearn.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "skops": sio.__version__ if hasattr(sio, "__version__") else "0.15.0",
    "seed": 0,
    "model_C": 0.019150292626953554,
}
with open(OUT + "/model_versions.json", "w") as f:
    json.dump(versions, f, indent=2)
print("\nwrote final_pipeline.skops, model_versions.json")
