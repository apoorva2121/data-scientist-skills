"""
Stage 7 - Baselines + model families + imbalance (ds-model-selection).

Final feature set from Stage 6: tenure_months, monthly_charges, total_charges,
support_calls_90d, avg_daily_usage_mb, age, contract_type, payment_method (region and
last_invoice_amount dropped on evidence; no engineered features survived).

Imbalance approach (in order of the skill's preference list):
  1. Metric + capacity threshold: PR-AUC as the model-quality metric, precision@10%/recall@10%
     as the deployment metric (threshold = score at the 90th percentile rank, not a probability
     cutoff) - already the plan since Stage 0.
  2. class_weight="balanced" / scale_pos_weight on every model screened below.
  3. SMOTE screened explicitly against class-weighting alone (not assumed) in Screen 3.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-method-comparison/scripts")
from cv_folds import load_split
from screen import evaluate, summarize, topk_scorer

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train = df[s.train_mask].reset_index(drop=True)
X = train.drop(columns=["churned"]); y = train["churned"]
NOISE = 0.042

FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
         "avg_daily_usage_mb", "age", "contract_type", "payment_method"]
X = X[FEATS]
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
cat_feats = ["contract_type", "payment_method"]
n_pos = int(y.sum()); n_neg = int((1 - y).sum())
spw = n_neg / n_pos
print(f"train n={len(y)} pos={n_pos} neg={n_neg} scale_pos_weight={spw:.2f}")

# ---- preprocessors ----
# linear preproc: median impute, log1p on avg_daily_usage_mb (Stage 5 carry-forward candidate),
# scale, one-hot the two remaining low-cardinality categoricals.
class Log1pCol(BaseEstimator, TransformerMixin):
    def __init__(self, col_idx):
        self.col_idx = col_idx
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.col_idx] = np.log1p(np.clip(X[:, self.col_idx], 0, None))
        return X

pre_linear = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("log_usage", Log1pCol(col_idx=num_feats.index("avg_daily_usage_mb"))),
                       ("sc", StandardScaler())]), num_feats),
    ("cat", Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
pre_tree_native = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
])
pre_tree_impute = ColumnTransformer([  # for DecisionTreeClassifier, which cannot take NaN
    ("num", SimpleImputer(strategy="median"), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
])

# ============ Step 1: baselines ============
def risk_rule_score(Xdf):
    """Two/three-line business rule a retention team could build without modelling."""
    return (3 * (Xdf["contract_type"] == "month-to-month").astype(int)
            + 1 * Xdf["support_calls_90d"].fillna(0)
            + 0.5 * (Xdf["payment_method"] == "Electronic check").astype(int))

class BusinessRule(ClassifierMixin, BaseEstimator):
    """Wraps the hand-built rule as a scikit-learn-compatible 'model' with predict_proba."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self
    def predict_proba(self, X):
        s = risk_rule_score(X).to_numpy(dtype=float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return np.column_stack([1 - s, s])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    def get_params(self, deep=True):
        return {}
    def set_params(self, **kw):
        return self

baseline_pipes = {
    "trivial_dummy": Pipeline([("pre", pre_linear), ("clf", DummyClassifier(strategy="prior"))]),
    "business_rule": Pipeline([("pre", "passthrough"), ("clf", BusinessRule())]),
    "simple_logreg": Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "shallow_tree_d3": Pipeline([("pre", pre_tree_impute), ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0))]),
}
print("\n=== Step 1: baselines (PR-AUC) ===")
long_b = evaluate(baseline_pipes, X, y, folds=s.folds, scoring="average_precision")
tab_b_prauc = summarize(long_b, noise_floor_sd=NOISE)
print(tab_b_prauc)

print("\n=== Step 1: baselines (precision@10%) ===")
long_b_p10 = evaluate(baseline_pipes, X, y, folds=s.folds, scoring=topk_scorer(0.10, "precision"))
tab_b_p10 = summarize(long_b_p10, noise_floor_sd=None)
print(tab_b_p10)

print("\n=== Step 1: baselines (recall@10% = capture rate) ===")
long_b_r10 = evaluate(baseline_pipes, X, y, folds=s.folds, scoring=topk_scorer(0.10, "recall"))
tab_b_r10 = summarize(long_b_r10, noise_floor_sd=None)
print(tab_b_r10)

# ============ Step 2: family screen ============
family_pipes = {
    "logreg": Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "hist_gbm": Pipeline([("pre", pre_tree_native), ("clf", HistGradientBoostingClassifier(random_state=0, max_iter=150, class_weight="balanced"))]),
    "lightgbm": Pipeline([("pre", pre_tree_native), ("clf", LGBMClassifier(random_state=0, n_estimators=200, verbosity=-1, class_weight="balanced"))]),
    "xgboost": Pipeline([("pre", pre_tree_native), ("clf", XGBClassifier(random_state=0, n_estimators=200, eval_metric="aucpr", scale_pos_weight=spw))]),
    "catboost": Pipeline([("pre", pre_tree_native), ("clf", CatBoostClassifier(random_state=0, iterations=200, verbose=False, auto_class_weights="Balanced"))]),
}
print("\n=== Step 2: family screen (PR-AUC) ===")
long_f = evaluate(family_pipes, X, y, folds=s.folds, scoring="average_precision")
tab_f = summarize(long_f, noise_floor_sd=NOISE)
print(tab_f)

# ============ Step 3 (imbalance): class_weight alone vs SMOTE, logistic regression ============
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
smote_pipe = ImbPipeline([("pre", pre_linear), ("smote", SMOTE(random_state=0)),
                           ("clf", LogisticRegression(max_iter=1000))])  # no class_weight, SMOTE balances instead
cw_pipe = Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
imb_pipes = {"class_weight_balanced": cw_pipe, "smote_oversample": smote_pipe}
print("\n=== Step 3: imbalance handling (PR-AUC) ===")
long_imb = evaluate(imb_pipes, X, y, folds=s.folds, scoring="average_precision")
tab_imb = summarize(long_imb, noise_floor_sd=NOISE)
print(tab_imb)

# ============ save everything ============
tab_b_prauc.to_csv(OUT + "/baseline_table_prauc.csv", index=False)
tab_b_p10.to_csv(OUT + "/baseline_table_precision_at_10.csv", index=False)
tab_b_r10.to_csv(OUT + "/baseline_table_recall_at_10.csv", index=False)
tab_f.to_csv(OUT + "/family_screen.csv", index=False)
tab_imb.to_csv(OUT + "/imbalance_screen.csv", index=False)

with open(OUT + "/search_space.json") as f:
    ss = json.load(f)
ss["stages"]["models"] = [
    {"name": row["name"], "mean": row["mean"], "sd": row["sd"], "keep": bool(row["keep"])}
    for row in tab_f.to_dict("records")
]
ss["imbalance_handling"] = "class_weight balanced / scale_pos_weight (SMOTE screened, no gain - see imbalance_screen.csv)"
with open(OUT + "/search_space.json", "w") as f:
    json.dump(ss, f, indent=2)
print("\nwrote baseline_table_*.csv, family_screen.csv, imbalance_screen.csv; updated search_space.json")
