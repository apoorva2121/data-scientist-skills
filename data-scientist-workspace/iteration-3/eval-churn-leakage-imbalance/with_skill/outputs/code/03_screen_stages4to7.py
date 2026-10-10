"""Stages 4-6 (missing values, outliers, features) + Stage 7 (baselines, model families, imbalance)
combined into one script - compact mode, time-boxed. Uses screen.py's evaluate/summarize framework
(ds-method-comparison/scripts/screen.py) and topk_scorer for the capacity-limited primary metric.
All fitting happens inside CV folds (non-negotiable #1); no test-set rows are touched here.
"""
import sys, json, time
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
import numpy as np, pandas as pd
from cv_folds import load_split
from screen import evaluate, summarize, topk_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve
import lightgbm as lgb
from catboost import CatBoostClassifier

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-churn-leakage-imbalance\with_skill\outputs"
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
       "avg_daily_usage_mb", "age", "last_invoice_amount"]
CAT = ["contract_type", "payment_method", "region"]
SKEWED = ["avg_daily_usage_mb"]  # EXTREME_MAX 73x p99 (column_audit.csv)

df = pd.read_csv(f"{OUT}/cleaned.csv")
s = load_split(f"{OUT}/split/split.csv", ids=df["customer_id"])
tr = df[s.train_mask].reset_index(drop=True)
X, y = tr[NUM + CAT], tr["churned"].to_numpy()
scorer = topk_scorer(0.10, "precision")

def winsorize(Xdf):
    Xdf = pd.DataFrame(Xdf, columns=NUM).copy()
    for c in SKEWED:
        lo, hi = Xdf[c].quantile(0.01), Xdf[c].quantile(0.99)
        Xdf[c] = Xdf[c].clip(lo, hi)
    return Xdf.to_numpy()

# ---- reference preprocessing per family / slot option ----
def linear_pipe(missing="median_indicator", outlier="none"):
    num_steps = []
    if outlier == "winsorize":
        num_steps.append(FunctionTransformer(winsorize))
    num_steps.append(SimpleImputer(strategy="median", add_indicator=(missing == "median_indicator")))
    num_steps.append(StandardScaler())
    num_pipe = make_pipeline(*num_steps)
    pre = ColumnTransformer([("num", num_pipe, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))])

def gbdt_pipe(missing="native_nan", outlier="none", model="lightgbm"):
    num_steps = []
    if outlier == "winsorize":
        num_steps.append(FunctionTransformer(winsorize))
    if missing == "median_indicator":
        num_steps.append(SimpleImputer(strategy="median", add_indicator=True))
    num_pipe = make_pipeline(*num_steps) if num_steps else FunctionTransformer(lambda x: x)
    pre = ColumnTransformer([("num", num_pipe, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    if model == "lightgbm":
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=30, class_weight="balanced", verbose=-1, random_state=0)
    else:
        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, auto_class_weights="Balanced",
                                  verbose=False, random_state=0)
    return Pipeline([("pre", pre), ("model", clf)])

print("=== Noise floor (precision@10%, RepeatedStratifiedKFold 5x3, reference linear pipeline) ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
scores = cross_val_score(linear_pipe(), X, y, cv=cv, scoring=scorer)
scores = scores.reshape(3, 5)  # repeats x folds
fold_sd = scores.std(ddof=1)
repeat_means = scores.mean(axis=1)
repeat_sd = repeat_means.std(ddof=1)
print(f"fold_sd={fold_sd:.4f} repeat_sd(noise_floor_sd)={repeat_sd:.4f} mean={scores.mean():.4f}")

print("\n=== Stage 4-5: missing values x outliers screen (2 proxies) ===")
pipes = {
    "linear__median_indicator__none": linear_pipe("median_indicator", "none"),
    "linear__median_indicator__winsorize": linear_pipe("median_indicator", "winsorize"),
    "gbdt__native_nan__none": gbdt_pipe("native_nan", "none"),
    "gbdt__native_nan__winsorize": gbdt_pipe("native_nan", "winsorize"),
    "gbdt__median_indicator__none": gbdt_pipe("median_indicator", "none"),
}
long = evaluate(pipes, X, y, folds=s.folds, scoring=scorer, seeds=range(2), seed_param="model__random_state")
tbl45 = summarize(long, noise_floor_sd=repeat_sd)
print(tbl45.to_string(index=False))
tbl45.to_csv(f"{OUT}/stage4_5_missing_outliers_screen.csv", index=False)

print("\n=== Stage 7 step 1: baselines ===")
base_pipes = {
    "trivial_prior": Pipeline([("pre", "passthrough"), ("model", DummyClassifier(strategy="prior"))]),
}
# business rule score: continuous risk proxy (month-to-month contract + support call volume), no fitting needed
class BusinessRule:
    _estimator_type = "classifier"
    classes_ = np.array([0, 1])
    def fit(self, X, y=None): return self
    def predict_proba(self, X):
        s_ = (X["contract_type"].eq("Month-to-month")).astype(float) * 2 + X["support_calls_90d"].astype(float)
        s_ = (s_ - s_.min()) / (s_.max() - s_.min() + 1e-9)
        return np.c_[1 - s_, s_]
    def get_params(self, deep=True): return {}
    def set_params(self, **kw): return self

class PassthroughPre:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None): return X

business_pipe = Pipeline([("pre", PassthroughPre()), ("model", BusinessRule())])
shallow_tree = Pipeline([("pre", ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                                                     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])),
                          ("model", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0))])
base_pipes["business_rule"] = business_pipe
base_pipes["simple_logistic"] = linear_pipe("median_indicator", "none")
base_pipes["shallow_tree_d3"] = shallow_tree
long_b = evaluate(base_pipes, X, y, folds=s.folds, scoring=scorer, seeds=range(2), seed_param=None)
tbl_base = summarize(long_b, noise_floor_sd=repeat_sd)
print(tbl_base.to_string(index=False))
tbl_base.to_csv(f"{OUT}/stage7_baselines.csv", index=False)

print("\n=== Stage 7 step 2-3: family screen (linear, lightgbm, catboost) ===")
fam_pipes = {
    "linear": linear_pipe("median_indicator", "winsorize"),
    "lightgbm": gbdt_pipe("native_nan", "none", "lightgbm"),
    "catboost": gbdt_pipe("native_nan", "none", "catboost"),
}
t0 = time.time()
long_f = evaluate(fam_pipes, X, y, folds=s.folds, scoring=scorer, seeds=range(2), seed_param="model__random_state")
tbl_fam = summarize(long_f, noise_floor_sd=repeat_sd)
print(tbl_fam.to_string(index=False))
print(f"family screen wall time: {time.time()-t0:.1f}s")
tbl_fam.to_csv(f"{OUT}/stage7_family_screen.csv", index=False)

print("\n=== Learning curve, top family (lightgbm) ===")
sizes, tr_sc, va_sc = learning_curve(fam_pipes["lightgbm"], X, y, cv=s.folds, scoring=scorer,
                                      train_sizes=[0.25, 0.5, 0.75, 1.0])
lc = pd.DataFrame({"train_size": sizes, "train_mean": tr_sc.mean(1), "val_mean": va_sc.mean(1)})
print(lc.to_string(index=False))
lc.to_csv(f"{OUT}/stage7_learning_curve.csv", index=False)

# ---- write search_space.json ----
def pack(tbl, applies):
    out = []
    for _, r in tbl.iterrows():
        out.append({"name": r["name"], "applies_to": applies.get(r["name"], ["linear", "gbdt"]),
                    "proxy_mean": round(float(r["mean"]), 4), "proxy_sd": round(float(r["sd"]), 4),
                    "keep": bool(r["keep"])})
    return out

space = {
    "cv": {"scheme": "StratifiedKFold(from split)", "n_splits": 5, "n_repeats": 1, "groups": None, "seed": 42},
    "primary_metric": "precision@0.10",
    "noise_floor_sd": round(float(repeat_sd), 4),
    "noise_floor_definition": "sd of the CV mean (repeat_sd) across 3 repeats of RepeatedStratifiedKFold(5), reference linear pipeline, precision@10% units",
    "fixed_steps": [
        "drop 150 exact duplicate rows",
        "contract_type spelling normalized to 3 levels",
        "total_charges: -999 -> NaN",
        "age: >120 -> NaN",
        "DROP churn_reason (leak, score 1.00)",
        "DROP account_status_code (leak, score 0.99)"
    ],
    "stages": {
        "missing_values": [
            {"name": "median_indicator", "applies_to": ["linear", "gbdt"]},
            {"name": "native_nan", "applies_to": ["gbdt"]},
        ],
        "outliers": [
            {"name": "none", "applies_to": ["linear", "gbdt"]},
            {"name": "winsorize_p01_p99", "applies_to": ["linear", "gbdt"]},
        ],
        "encoding": [
            {"name": "onehot", "applies_to": ["linear", "gbdt"]},
        ],
        "models": pack(tbl_fam, {"linear": ["linear"], "lightgbm": ["gbdt"], "catboost": ["gbdt"]}),
    },
}
with open(f"{OUT}/search_space.json", "w") as f:
    json.dump(space, f, indent=2)
print("\nwritten search_space.json, noise_floor_sd=", round(float(repeat_sd), 4))
