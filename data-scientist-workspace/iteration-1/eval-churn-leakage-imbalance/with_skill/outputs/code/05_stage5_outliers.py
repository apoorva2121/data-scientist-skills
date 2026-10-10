"""
Stage 5 - Outliers (ds-outliers).

Profile (outlier_profile.csv, train n=3900):
  avg_daily_usage_mb: skew=19.57, 194 rows beyond 1.5 IQR, 15 rows > 10x p99 (max 4196 vs p99
    58). Manually inspected the 15 extreme rows: values are non-round (672, 711, 1155, ... 4196),
    no x10/x100/x1000 unit-shift pattern found, and 0/15 churned - consistent with chance under
    the 8.67% base rate (binomial P(0|n=15)~0.26), not a distinct signal cluster. No domain rule
    makes >4000 MB/day impossible for a heavy household/business connection. Verdict: genuine
    heavy tail, not an error - transform candidate for the linear proxy, leave as-is for trees.
  total_charges: skew=0.75 (below the "heavy tail" cutoff of ~1), 30 rows beyond 1.5 IQR, 0/30
    (or fewer) churned among them - fully explained by the known tenure-churn relationship
    (extreme total_charges rows are long-tenure customers, and long tenure already predicts low
    churn). Verdict: genuine, no special treatment.
  last_invoice_amount: only 2 rows beyond a 3.5-MAD threshold, both high-monthly-charge customers
    (consistent with its 0.97 correlation to monthly_charges). n=2 is far below the skill's
    "roughly 10 or more" bar for treating a lift as signal. Verdict: undecided/isolated, default
    keep, no action.
  tenure_months, monthly_charges, age: "no extremes" per the profile script.

Screens: none (reference) vs log1p(avg_daily_usage_mb) vs Winsorizer(0.01, 0.99) on
avg_daily_usage_mb + total_charges, linear proxy only (trees are split-based and insensitive to
monotonic transforms / capping - screened once as a reference, not re-screened per candidate).
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-outliers/scripts")
from cv_folds import load_split
from outlier_transformers import Winsorizer

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train = df[s.train_mask].reset_index(drop=True)
X = train.drop(columns=["churned"]); y = train["churned"]

num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
             "avg_daily_usage_mb", "age", "last_invoice_amount"]
cat_feats = ["contract_type", "payment_method", "region"]

results = {}

def linear_pipe(num_transform_step=None):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if num_transform_step is not None:
        steps.append(num_transform_step)
    steps.append(("sc", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(steps), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def score(name, pipe):
    sc = cross_val_score(pipe, X, y, cv=s.folds, scoring="average_precision", n_jobs=-1)
    results[name] = [round(float(sc.mean()), 4), round(float(sc.std(ddof=1)), 4)]
    print(f"{name:24s} linear PR-AUC={sc.mean():.4f}+/-{sc.std(ddof=1):.4f}")

# candidate: none
score("none", linear_pipe())

# candidate: log1p on avg_daily_usage_mb only -> apply via FunctionTransformer on the whole
# numeric block would log1p everything; instead build a dedicated ColumnTransformer for this one.
pre_log = ColumnTransformer([
    ("usage_log", Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, validate=True)), ("sc", StandardScaler())]), ["avg_daily_usage_mb"]),
    ("other_num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]),
     [c for c in num_feats if c != "avg_daily_usage_mb"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pipe_log = Pipeline([("pre", pre_log), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
score("log1p_usage", pipe_log)

# candidate: winsorize avg_daily_usage_mb and total_charges at (0.01, 0.99), fit in-fold
pre_wins = ColumnTransformer([
    ("wins", Pipeline([("imp", SimpleImputer(strategy="median")), ("wz", Winsorizer(0.01, 0.99)), ("sc", StandardScaler())]),
     ["avg_daily_usage_mb", "total_charges"]),
    ("other_num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]),
     [c for c in num_feats if c not in ("avg_daily_usage_mb", "total_charges")]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pipe_wins = Pipeline([("pre", pre_wins), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
score("winsorize_p01_p99", pipe_wins)

# tree reference (no transform - trees are split-based, insensitive to monotonic transforms)
pre_tree = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
])
pipe_tree = Pipeline([("pre", pre_tree), ("clf", HistGradientBoostingClassifier(random_state=0, max_iter=150))])
sc_tree = cross_val_score(pipe_tree, X, y, cv=s.folds, scoring="average_precision", n_jobs=-1)
results["none_gbdt"] = [round(float(sc_tree.mean()), 4), round(float(sc_tree.std(ddof=1)), 4)]
print(f"{'none (gbdt reference)':24s} gbdt PR-AUC={sc_tree.mean():.4f}+/-{sc_tree.std(ddof=1):.4f}")

print("\n", json.dumps(results, indent=2))

with open(OUT + "/search_space.json") as f:
    ss = json.load(f)
ss["stages"]["outliers"] = [
    {"name": "none", "applies_to": ["linear", "gbdt"], "proxy_scores": {"linear": results["none"], "gbdt": results["none_gbdt"]}},
    {"name": "log1p_usage", "applies_to": ["linear"], "proxy_scores": {"linear": results["log1p_usage"]}},
]
with open(OUT + "/search_space.json", "w") as f:
    json.dump(ss, f, indent=2)
print("updated search_space.json (outliers)")
