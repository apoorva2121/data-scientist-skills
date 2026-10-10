"""
Stage 4 - Missing values (ds-missing-values).

Diagnosis (missing_profile.csv, train-only):
  last_invoice_amount  18.0% missing  predictable_from_obs=0.68  target_signal=0.568  -> informative + MAR-ish
  total_charges         9.95% missing predictable_from_obs=0.49  target_signal=0.51   -> MCAR-like
  age                    0.97% missing predictable_from_obs=0.46  target_signal=0.50   -> MCAR-like

Screens candidates with two proxies (regularised linear, small GBDT) on the Stage 2 CV folds
(the fixed 5 folds from split/split.csv, not a fresh CV) so results are comparable across
stages, per ds-method-comparison's carry-forward convention.
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
from sklearn.model_selection import cross_val_score

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
from cv_folds import load_split

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train = df[s.train_mask].reset_index(drop=True)  # matches s.folds positional indices
X_all = train.drop(columns=["churned"])
y = train["churned"]

num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
             "avg_daily_usage_mb", "age", "last_invoice_amount"]
cat_feats = ["contract_type", "payment_method", "region"]
miss_cols = ["total_charges", "age", "last_invoice_amount"]

results = {}

def run(name, pre_linear, pre_tree):
    lin = Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    tree = Pipeline([("pre", pre_tree), ("clf", HistGradientBoostingClassifier(random_state=0, max_iter=150))])
    sc_lin = cross_val_score(lin, X_all, y, cv=s.folds, scoring="average_precision", n_jobs=-1)
    sc_tree = cross_val_score(tree, X_all, y, cv=s.folds, scoring="average_precision", n_jobs=-1)
    results[name] = {
        "linear": [round(float(sc_lin.mean()), 4), round(float(sc_lin.std(ddof=1)), 4)],
        "gbdt":   [round(float(sc_tree.mean()), 4), round(float(sc_tree.std(ddof=1)), 4)],
    }
    print(f"{name:22s} linear={sc_lin.mean():.4f}+/-{sc_lin.std(ddof=1):.4f}   gbdt={sc_tree.mean():.4f}+/-{sc_tree.std(ddof=1):.4f}")

# --- candidate 1: median impute, no indicator ---
pre_lin_median = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pre_tree_median = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
run("median_no_indicator", pre_lin_median, pre_tree_median)

# --- candidate 2: median impute + missing indicator on all 3 missing columns ---
pre_lin_ind = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pre_tree_ind = ColumnTransformer([
    ("num", SimpleImputer(strategy="median", add_indicator=True), num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
run("median_indicator_all", pre_lin_ind, pre_tree_ind)

# --- candidate 3: median + indicator only on the informative column (last_invoice_amount) ---
pre_lin_ind1 = ColumnTransformer([
    ("num_plain", Pipeline([("imp", SimpleImputer(strategy="median"))]), ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]),
    ("num_ind", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]), ["last_invoice_amount"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
# note: ColumnTransformer output order differs from num_feats but StandardScaler must come after concat; simplify: scale within each numeric block
pre_lin_ind1 = ColumnTransformer([
    ("num_plain", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]),
    ("num_ind", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), ["last_invoice_amount"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pre_tree_ind1 = ColumnTransformer([
    ("num_plain", SimpleImputer(strategy="median"), ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]),
    ("num_ind", SimpleImputer(strategy="median", add_indicator=True), ["last_invoice_amount"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
run("median_indicator_informative_only", pre_lin_ind1, pre_tree_ind1)

# --- candidate 4: native NaN for the tree (no imputation at all on numerics); linear still needs median (can't handle NaN) ---
class PassthroughNaN:
    pass
pre_tree_native = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
tree_native = Pipeline([("pre", pre_tree_native), ("clf", HistGradientBoostingClassifier(random_state=0, max_iter=150))])
sc_tree_native = cross_val_score(tree_native, X_all, y, cv=s.folds, scoring="average_precision", n_jobs=-1)
results["native_nan_gbdt_only"] = {"gbdt": [round(float(sc_tree_native.mean()), 4), round(float(sc_tree_native.std(ddof=1)), 4)]}
print(f"{'native_nan_gbdt_only':22s} gbdt={sc_tree_native.mean():.4f}+/-{sc_tree_native.std(ddof=1):.4f}  (linear: N/A, sklearn LogisticRegression cannot take NaN)")

print("\nAll candidate scores:")
print(json.dumps(results, indent=2))

# --- update search_space.json ---
with open(OUT + "/search_space.json") as f:
    ss = json.load(f)
ss["stages"]["missing_values"] = [
    {"name": "median_no_indicator", "applies_to": ["linear", "gbdt"], "proxy_scores": results["median_no_indicator"]},
    {"name": "median_indicator_informative_only", "applies_to": ["linear", "gbdt"], "proxy_scores": results["median_indicator_informative_only"]},
    {"name": "native_nan", "applies_to": ["gbdt"], "proxy_scores": results["native_nan_gbdt_only"]},
]
with open(OUT + "/search_space.json", "w") as f:
    json.dump(ss, f, indent=2)
print("\nupdated search_space.json (missing_values)")
