"""
03_tune_and_finalize.py

The baseline comparison (02_model_comparison.py) showed logistic
regression beating default-hyperparameter tree ensembles on 5-fold CV
PR-AUC, on this small (~4,800 training row), noisy, weakly-linear
dataset. Before concluding logistic regression is "the best model", we
give every family a fair chance by tuning hyperparameters with Optuna
(3-fold CV PR-AUC as the objective, on the training split only -- the
test split is never touched during tuning).

Whichever family wins on tuned CV PR-AUC is refit on the full training
split and evaluated exactly once on the held-out test split. That single
held-out evaluation is what's reported to the client as "how good the
model really is".
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss,
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = Path(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-churn-leakage-imbalance\without_skill\outputs")
RANDOM_STATE = 42
K_FRAC = 0.10
N_TRIALS = 15
N_TUNE_FOLDS = 3
# This machine appears to be shared with other heavy, unrelated processes;
# n_jobs=1 avoids thread-oversubscription contention and is empirically
# much faster in practice here than n_jobs=-1 for this small dataset.
N_JOBS_MODEL = 1

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
    return precision_at_k, recall_at_k, lift_at_k, k


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


def cv_pr_auc(pipe_builder, X, y, n_folds=N_TUNE_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        pipe = pipe_builder()
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(average_precision_score(y[va_idx], p))
    return float(np.mean(scores))


def make_objective(family, X, y, scale_pos_weight):
    def objective(trial):
        if family == "logistic_regression":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            clf = LogisticRegression(
                C=C, penalty=penalty, solver="liblinear",
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
            )
        elif family == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 100, 300, step=50)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            max_features = trial.suggest_float("max_features", 0.3, 1.0)
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL,
            )
        elif family == "lightgbm":
            clf = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                num_leaves=trial.suggest_int("num_leaves", 7, 63),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1, n_jobs=N_JOBS_MODEL,
            )
        elif family == "xgboost":
            clf = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL, verbosity=0,
            )
        else:
            raise ValueError(family)

        def build():
            return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

        return cv_pr_auc(build, X, y)
    return objective


def main():
    train_df = pd.read_csv(OUT_DIR / "train_split.csv")
    test_df = pd.read_csv(OUT_DIR / "test_split.csv")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET].values
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET].values

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    families = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
    best_per_family = {}

    for family in families:
        t0 = time.time()
        print(f"--- tuning {family} ({N_TRIALS} trials x {N_TUNE_FOLDS}-fold CV) ---", flush=True)
        sampler = TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(make_objective(family, X_train, y_train, scale_pos_weight),
                        n_trials=N_TRIALS, show_progress_bar=False)
        best_per_family[family] = {"pr_auc_cv": study.best_value, "params": study.best_params}
        elapsed = time.time() - t0
        print(f"{family}: best tuned 3-fold CV PR-AUC = {study.best_value:.4f}  "
              f"params={study.best_params}  ({elapsed:.1f}s)", flush=True)
        # write partial results after every family so progress survives even
        # if a later family is interrupted
        with open(OUT_DIR / "tuning_results.json", "w") as f:
            json.dump(best_per_family, f, indent=2)

    with open(OUT_DIR / "tuning_results.json", "w") as f:
        json.dump(best_per_family, f, indent=2)

    best_family = max(best_per_family, key=lambda k: best_per_family[k]["pr_auc_cv"])
    best_params = best_per_family[best_family]["params"]
    print(f"\nSelected final family: {best_family} (tuned CV PR-AUC={best_per_family[best_family]['pr_auc_cv']:.4f})")

    # rebuild the winning classifier with its best params
    if best_family == "logistic_regression":
        clf = LogisticRegression(
            C=best_params["C"], penalty=best_params["penalty"], solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        )
    elif best_family == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"], max_features=best_params["max_features"],
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL,
        )
    elif best_family == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=best_params["n_estimators"], num_leaves=best_params["num_leaves"],
            max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"],
            min_child_samples=best_params["min_child_samples"], reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"], subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1,
        )
    else:  # xgboost
        clf = XGBClassifier(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"], min_child_weight=best_params["min_child_weight"],
            reg_alpha=best_params["reg_alpha"], reg_lambda=best_params["reg_lambda"],
            subsample=best_params["subsample"], colsample_bytree=best_params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL, verbosity=0,
        )

    final_pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    final_pipe.fit(X_train, y_train)

    # ---- single, final held-out evaluation ----
    test_scores = final_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    p10, r10, lift10, k = precision_recall_at_k(y_test, test_scores, K_FRAC)
    brier = brier_score_loss(y_test, test_scores)

    threshold_at_k = np.sort(test_scores)[::-1][k - 1]
    y_pred_at_k = (test_scores >= threshold_at_k).astype(int)
    cm = confusion_matrix(y_test, y_pred_at_k)

    metrics = {
        "final_model_family": best_family,
        "final_model_params": best_params,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_churn_rate": float(y_test.mean()),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_brier_score": float(brier),
        "k_frac": K_FRAC,
        "k_customers_in_test": int(k),
        "test_precision_at_10pct": float(p10),
        "test_recall_at_10pct_capture_rate": float(r10),
        "test_lift_at_10pct": float(lift10),
        "score_threshold_for_top_10pct": float(threshold_at_k),
        "confusion_matrix_at_top10pct_threshold": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }
    print("\n=== FINAL HELD-OUT TEST METRICS ===")
    print(json.dumps(metrics, indent=2))

    with open(OUT_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(final_pipe, OUT_DIR / "final_model_pipeline.joblib")

    # ---- plots ----
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr, tpr, _ = roc_curve(y_test, test_scores)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (test)"); axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, test_scores)
    axes[1].plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4, label=f"base rate={y_test.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (test)"); axes[1].legend()

    order = np.argsort(-test_scores)
    y_sorted = y_test[order]
    n = len(y_test)
    fracs = np.arange(1, n + 1) / n
    cum_capture = np.cumsum(y_sorted) / y_sorted.sum()
    axes[2].plot(fracs * 100, cum_capture * 100, label="model")
    axes[2].plot([0, 100], [0, 100], "k--", alpha=0.4, label="random")
    axes[2].axvline(10, color="red", linestyle=":", alpha=0.6, label="10% call capacity")
    axes[2].set_xlabel("% of customers contacted (ranked by predicted risk)")
    axes[2].set_ylabel("% of actual churners captured")
    axes[2].set_title("Cumulative Gains Chart (test)"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "evaluation_plots.png", dpi=130)
    print(f"\nSaved plots to {OUT_DIR / 'evaluation_plots.png'}")


if __name__ == "__main__":
    main()
