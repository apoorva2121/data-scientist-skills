"""Stage 7 - learning curve for the top 2 families (logreg, catboost) to judge learnability."""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
from cv_folds import load_split

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
train = df[s.train_mask].reset_index(drop=True)
FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
         "avg_daily_usage_mb", "age", "contract_type", "payment_method"]
X = train[FEATS]; y = train["churned"]
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
cat_feats = ["contract_type", "payment_method"]

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
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
pre_tree = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
])

logreg_pipe = Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
catboost_pipe = Pipeline([("pre", pre_tree), ("clf", CatBoostClassifier(random_state=0, iterations=200, verbose=False, auto_class_weights="Balanced"))])

rows = []
for name, pipe in [("logreg", logreg_pipe), ("catboost", catboost_pipe)]:
    sizes, tr_scores, va_scores = learning_curve(
        pipe, X, y, cv=s.folds, scoring="average_precision",
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], shuffle=False)
    for sz, tr, va in zip(sizes, tr_scores, va_scores):
        rows.append({"model": name, "train_size": int(sz), "train_mean": round(tr.mean(), 4),
                      "val_mean": round(va.mean(), 4), "val_sd": round(va.std(ddof=1), 4)})
    print(f"\n{name}:")
    for r in rows:
        if r["model"] == name:
            print(f"  n={r['train_size']:4d}  train={r['train_mean']:.4f}  val={r['val_mean']:.4f} +/- {r['val_sd']:.4f}")

pd.DataFrame(rows).to_csv(OUT + "/learning_curve.csv", index=False)
print("\nwrote learning_curve.csv")
