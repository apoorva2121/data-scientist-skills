"""
Stage 2 - Validation design + carve test (ds-validation-design).

Deployment question: rows are independent customers (no repeated-entity structure after
dedup at Stage 1); no reliable snapshot/prediction-moment timestamp exists (signup_date is
decorrelated from tenure_months, r=0.019, and there is no extraction-date column); churn
rate by signup-month shows no monotonic drift (fluctuates 4.7%-12.8% with no trend). ->
scheme = stratified (independent rows, no drift, binary classification).
Rejected: time-based split (no trustworthy time index to split on; would silently pick an
arbitrary, meaningless ordering); group split (no group/entity column identified at Stage 1).

This script: (a) confirms the Stage-2 split already carved with make_split.py, (b) runs the
sanity checks (target balance per part, adversarial validation), (c) computes the test-set
precision estimate, (d) measures the noise floor with one simple pipeline under repeated
stratified CV on the train part only.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
from cv_folds import load_split, verify_test_untouched

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])

print("verify_test_untouched:", verify_test_untouched(OUT + "/split"))
print("train:", s.train_mask.sum(), "holdout:", s.holdout_mask.sum(), "test:", s.test_mask.sum())

# --- test-set precision estimate ---
n_test = int(s.test_mask.sum())
p_test = df.loc[s.test_mask, "churned"].mean()
se_prop = np.sqrt(p_test * (1 - p_test) / n_test)
print(f"test set: n={n_test}, base rate={p_test:.4f}")
print(f"precision@10% test SE (n~{int(n_test*0.10)} calls, assume p~0.30): "
      f"{np.sqrt(0.30*0.70/int(n_test*0.10)):.4f} -> 95% CI +/- {1.96*np.sqrt(0.30*0.70/int(n_test*0.10)):.4f}")
n1 = int(df.loc[s.test_mask, "churned"].sum()); n0 = n_test - n1
# Hanley-McNeil-style SE for AUC-type metric, rough guide
se_auc = np.sqrt((0.7 * 0.3 + (n1 - 1) * (0.7 / (2 - 0.7) - 0.7 ** 2) + (n0 - 1) * (2 * 0.7 ** 2 / (1 + 0.7) - 0.7 ** 2)) / (n1 * n0))
print(f"test set positives n1={n1}, negatives n0={n0}; illustrative AUC SE at AUC~0.70: {se_auc:.4f}")

# --- feature set for noise-floor pipeline only (NOT the final feature list) ---
# Excludes ID, date, and the two Stage-1-presumed-leak columns (formally confirmed Stage 3).
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
             "avg_daily_usage_mb", "age", "last_invoice_amount"]
cat_feats = ["contract_type", "payment_method", "region"]
drop_feats = ["customer_id", "signup_date", "churn_reason", "account_status_code", "churned"]

X = df[num_feats + cat_feats]
y = df["churned"]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_feats),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
simple_pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

X_train, y_train = X[s.train_mask], y[s.train_mask]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
scores = cross_val_score(simple_pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
scores_mat = scores.reshape(5, 5)  # repeats x splits (sklearn order: repeat-major)
fold_sd = float(scores.std(ddof=1))
repeat_sd = float(scores_mat.mean(axis=1).std(ddof=1))
print(f"noise floor (simple LogisticRegression, PR-AUC): mean={scores.mean():.4f} fold_sd={fold_sd:.4f} repeat_sd={repeat_sd:.4f}")

# --- adversarial validation: train vs test ---
Xadv = pd.concat([X[s.train_mask], X[s.test_mask]], axis=0)
yadv = np.r_[np.zeros(s.train_mask.sum()), np.ones(s.test_mask.sum())]
pre_adv = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feats),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
adv_pipe = Pipeline([("pre", pre_adv), ("clf", HistGradientBoostingClassifier(random_state=0))])
adv_auc = cross_val_score(adv_pipe, Xadv, yadv, cv=5, scoring="roc_auc", n_jobs=-1).mean()
print(f"adversarial validation AUC (train vs test, should be ~0.5): {adv_auc:.4f}")

# --- write search_space.json skeleton with noise floor ---
search_space = {
    "cv": {"scheme": "StratifiedKFold-on-fixed-folds", "n_splits": 5, "seed": 42,
           "note": "folds loaded from split/split.csv via cv_folds.load_split; noise floor measured with RepeatedStratifiedKFold(5x5) on train only"},
    "primary_metric": "average_precision",
    "deployment_metric": "precision_at_10pct / recall_at_10pct",
    "noise_floor_sd": {"fold_sd": round(fold_sd, 4), "repeat_sd": round(repeat_sd, 4)},
    "noise_floor_basis": "simple LogisticRegression on 7 numeric + 3 categorical honest features, median/most-frequent impute, PR-AUC",
    "adversarial_validation_auc_train_vs_test": round(float(adv_auc), 4),
    "fixed_steps": [
        "drop 150 exact duplicate rows",
        "total_charges: -999 -> NaN",
        "age: {0, 250} -> NaN",
        "contract_type: canonicalize spelling/case/whitespace to 3 levels",
        "drop churn_reason, account_status_code from features (presumed leak, confirmed Stage 3)"
    ],
    "stages": {"missing_values": [], "outliers": [], "features": [], "models": []}
}
with open(OUT + "/search_space.json", "w") as f:
    json.dump(search_space, f, indent=2)
print("wrote search_space.json")
