"""
02_model_comparison.py

Compares four candidate model families on the cleaned churn dataset using
5-fold stratified cross-validation on a training split (a separate test
split is held out untouched until the final script).

Business framing: the retention team can only call ~10% of the customer
base per month. That makes this fundamentally a RANKING problem, not a
plain classification problem, so alongside the usual discrimination
metrics (ROC-AUC, PR-AUC) we score every model on Precision@10% (of the
customers the model would flag as its top decile, what fraction actually
churn) and Recall@10% / capture rate (of all customers who actually
churn, what fraction are caught in that top decile).

Given the class imbalance (~8.7% churn), PR-AUC and Precision/Recall@10%
are treated as the primary metrics; ROC-AUC is reported for reference
but is less informative here since it is dominated by the (uninteresting)
majority class.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

OUT_DIR = Path(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-churn-leakage-imbalance\without_skill\outputs")
DATA_PATH = OUT_DIR / "churn_clean.csv"
RANDOM_STATE = 42
K_FRAC = 0.10  # retention team capacity

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "age", "support_calls_90d", "avg_daily_usage_mb",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "region"]
TARGET = "churned"


def precision_recall_at_k(y_true, y_score, k_frac=K_FRAC):
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    y_true = np.asarray(y_true)
    n_pos_total = y_true.sum()
    n_pos_in_top_k = y_true[top_k_idx].sum()
    precision_at_k = n_pos_in_top_k / k
    recall_at_k = n_pos_in_top_k / n_pos_total if n_pos_total > 0 else np.nan
    base_rate = n_pos_total / n
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else np.nan
    return precision_at_k, recall_at_k, lift_at_k


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def get_models(scale_pos_weight):
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=-1, num_leaves=15, learning_rate=0.05,
            is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)} (churn rate {y_train.mean():.4f}), "
          f"Test size: {len(X_test)} (churn rate {y_test.mean():.4f})")

    # save the split indices/data so later scripts reuse the *exact* same split
    X_train.assign(churned=y_train).to_csv(OUT_DIR / "train_split.csv", index=False)
    X_test.assign(churned=y_test).to_csv(OUT_DIR / "test_split.csv", index=False)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    preprocessor = build_preprocessor()
    models = get_models(scale_pos_weight)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        fold_metrics = {"pr_auc": [], "roc_auc": [], "precision_at_10": [],
                         "recall_at_10": [], "lift_at_10": []}
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            pipe.fit(X_tr, y_tr)
            scores = pipe.predict_proba(X_va)[:, 1]
            fold_metrics["pr_auc"].append(average_precision_score(y_va, scores))
            fold_metrics["roc_auc"].append(roc_auc_score(y_va, scores))
            p10, r10, l10 = precision_recall_at_k(y_va, scores)
            fold_metrics["precision_at_10"].append(p10)
            fold_metrics["recall_at_10"].append(r10)
            fold_metrics["lift_at_10"].append(l10)

        row = {"model": name}
        for metric, values in fold_metrics.items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        results.append(row)
        print(f"\n{name}:")
        for metric in fold_metrics:
            print(f"  {metric}: {np.mean(fold_metrics[metric]):.4f} +/- {np.std(fold_metrics[metric]):.4f}")

    results_df = pd.DataFrame(results).sort_values("pr_auc_mean", ascending=False)
    results_df.to_csv(OUT_DIR / "cv_model_comparison.csv", index=False)
    print("\n=== Ranked by mean CV PR-AUC ===")
    print(results_df[["model", "pr_auc_mean", "roc_auc_mean", "precision_at_10_mean", "recall_at_10_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
