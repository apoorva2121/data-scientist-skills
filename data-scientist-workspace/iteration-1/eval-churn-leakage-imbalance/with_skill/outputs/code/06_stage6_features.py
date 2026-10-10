"""
Stage 6 - Feature engineering (ds-feature-engineering).

Builds candidate engineered features from Stage 3 EDA notes, screens them (with and without,
per group) using ds-method-comparison's screen.evaluate/summarize on the fixed Stage 2 folds,
and screens a region encoding choice (one-hot vs target-encoding vs drop). Also resolves the
Stage 3/4 open question on last_invoice_amount via an explicit ablation.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-method-comparison/scripts")
from cv_folds import load_split
from screen import evaluate, summarize

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train = df[s.train_mask].reset_index(drop=True)
X_raw = train.drop(columns=["churned"]); y = train["churned"]
NOISE = 0.042  # fold_sd from search_space.json (Stage 2)

# --- engineered feature builder (pure arithmetic / fixed domain thresholds, stateless) ---
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds engineered columns. Thresholds are fixed domain constants chosen from Stage-3
    train-only EDA (month-to-month + short tenure segment; the 3-call threshold where churn
    rate visibly steps up), not statistics re-estimated per fold -> stateless, same rule as
    the ledger's 'charges_per_month' example."""
    def __init__(self, use_interaction=True, use_freq_caller=True, use_residual=True):
        self.use_interaction = use_interaction
        self.use_freq_caller = use_freq_caller
        self.use_residual = use_residual

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.use_interaction:
            X["high_risk_contract_tenure"] = ((X["contract_type"] == "month-to-month") & (X["tenure_months"] <= 8)).astype(int)
        if self.use_freq_caller:
            X["frequent_caller"] = (X["support_calls_90d"] >= 3).astype(int)
        if self.use_residual:
            X["total_charges_residual"] = X["total_charges"] - X["tenure_months"] * X["monthly_charges"]
        return X

num_base = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
cat_feats = ["contract_type", "payment_method", "region"]
eng_cols_all = ["high_risk_contract_tenure", "frequent_caller", "total_charges_residual"]

def make_pipe(model_family, use_engineered=True, use_last_invoice=True, region_encoding="onehot",
              use_interaction=True, use_freq_caller=True, use_residual=True):
    num_feats = list(num_base)
    if use_last_invoice:
        num_feats = num_feats + ["last_invoice_amount"]
    eng_cols = []
    if use_engineered:
        if use_interaction: eng_cols.append("high_risk_contract_tenure")
        if use_freq_caller: eng_cols.append("frequent_caller")
        if use_residual: eng_cols.append("total_charges_residual")

    if region_encoding == "onehot":
        cat_transform = ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore",
                                                                  sparse_output=(model_family == "linear")))]),
                          cat_feats)
        cat_blocks = [cat_transform]
    elif region_encoding == "targetenc":
        cat_transform_low = ("cat_low", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                    ("oh", OneHotEncoder(handle_unknown="ignore",
                                                                          sparse_output=(model_family == "linear")))]),
                              ["contract_type", "payment_method"])
        cat_transform_te = ("region_te", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                     ("te", TargetEncoder(random_state=0))]), ["region"])
        cat_blocks = [cat_transform_low, cat_transform_te]
    elif region_encoding == "drop":
        cat_transform = ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore",
                                                                  sparse_output=(model_family == "linear")))]),
                          ["contract_type", "payment_method"])
        cat_blocks = [cat_transform]
    else:
        raise ValueError(region_encoding)

    num_transform_steps = [("imp", SimpleImputer(strategy="median", add_indicator=use_last_invoice))]
    if model_family == "linear":
        num_transform_steps.append(("sc", StandardScaler()))
    num_block = ("num", Pipeline(num_transform_steps), num_feats)

    blocks = [num_block] + cat_blocks
    if eng_cols:
        # total_charges_residual inherits NaN from total_charges (9.95% missing); the two flag
        # columns are never missing. Median-impute the block (no-op for the flags).
        eng_block = ("eng", SimpleImputer(strategy="median"), eng_cols)
        blocks.append(eng_block)

    pre = ColumnTransformer(blocks)
    builder = FeatureBuilder(use_interaction=use_interaction, use_freq_caller=use_freq_caller, use_residual=use_residual) if eng_cols else "passthrough"
    if model_family == "linear":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=150)
    return Pipeline([("build", builder), ("pre", pre), ("clf", clf)])

# ============ Screen 1: baseline vs +engineered, both families ============
pipes = {
    "linear_baseline": make_pipe("linear", use_engineered=False),
    "linear_plus_engineered": make_pipe("linear", use_engineered=True),
    "gbdt_baseline": make_pipe("gbdt", use_engineered=False),
    "gbdt_plus_engineered": make_pipe("gbdt", use_engineered=True),
}
print("=== Screen 1: baseline vs +engineered ===")
long1 = evaluate(pipes, X_raw, y, folds=s.folds, scoring="average_precision")
tab1 = summarize(long1, noise_floor_sd=NOISE)
print(tab1)

# ============ Screen 2: ablation of each engineered group (linear proxy) ============
pipes2 = {
    "all_engineered": make_pipe("linear", use_engineered=True),
    "minus_interaction": make_pipe("linear", use_engineered=True, use_interaction=False),
    "minus_freq_caller": make_pipe("linear", use_engineered=True, use_freq_caller=False),
    "minus_residual": make_pipe("linear", use_engineered=True, use_residual=False),
}
print("\n=== Screen 2: ablation per engineered group (linear) ===")
long2 = evaluate(pipes2, X_raw, y, folds=s.folds, scoring="average_precision")
tab2 = summarize(long2, noise_floor_sd=NOISE)
print(tab2)

# ============ Screen 3: last_invoice_amount ablation (both families) ============
pipes3 = {
    "linear_with_lastinv": make_pipe("linear", use_engineered=True, use_last_invoice=True),
    "linear_without_lastinv": make_pipe("linear", use_engineered=True, use_last_invoice=False),
    "gbdt_with_lastinv": make_pipe("gbdt", use_engineered=True, use_last_invoice=True),
    "gbdt_without_lastinv": make_pipe("gbdt", use_engineered=True, use_last_invoice=False),
}
print("\n=== Screen 3: last_invoice_amount ablation ===")
long3 = evaluate(pipes3, X_raw, y, folds=s.folds, scoring="average_precision")
tab3 = summarize(long3, noise_floor_sd=NOISE)
print(tab3)

# ============ Screen 4: region encoding (both families) ============
pipes4 = {
    "linear_region_onehot": make_pipe("linear", use_engineered=True, region_encoding="onehot"),
    "linear_region_targetenc": make_pipe("linear", use_engineered=True, region_encoding="targetenc"),
    "linear_region_drop": make_pipe("linear", use_engineered=True, region_encoding="drop"),
    "gbdt_region_onehot": make_pipe("gbdt", use_engineered=True, region_encoding="onehot"),
    "gbdt_region_targetenc": make_pipe("gbdt", use_engineered=True, region_encoding="targetenc"),
    "gbdt_region_drop": make_pipe("gbdt", use_engineered=True, region_encoding="drop"),
}
print("\n=== Screen 4: region encoding ===")
long4 = evaluate(pipes4, X_raw, y, folds=s.folds, scoring="average_precision")
tab4 = summarize(long4, noise_floor_sd=NOISE)
print(tab4)

# save all screening tables
tab1.to_csv(OUT + "/feature_screen_baseline_vs_engineered.csv", index=False)
tab2.to_csv(OUT + "/feature_screen_ablation.csv", index=False)
tab3.to_csv(OUT + "/feature_screen_last_invoice.csv", index=False)
tab4.to_csv(OUT + "/feature_screen_region_encoding.csv", index=False)
print("\nwrote feature_screen_*.csv")
