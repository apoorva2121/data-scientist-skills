"""
04_explain_drivers.py

Loads the final tuned pipeline (saved by 03_tune_and_finalize.py) and
explains what drives its churn predictions using two independent methods
that should agree if the "drivers" are real signal and not an artifact
of one particular importance metric:

  1. SHAP values (on the held-out test split) -- gives both magnitude
     and direction of each feature's contribution.
  2. Permutation importance (on the held-out test split, scored on
     PR-AUC) -- a model-agnostic sanity check: how much does shuffling
     a feature hurt ranking quality.

Outputs: feature_importance_shap.csv, feature_importance_permutation.csv,
shap_summary.png
"""

import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

OUT_DIR = Path(r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-churn-leakage-imbalance\without_skill\outputs")

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "age", "support_calls_90d", "avg_daily_usage_mb",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "region"]
TARGET = "churned"
RANDOM_STATE = 42


def main():
    # joblib.load executes a pickle; safe here only because this file was
    # written by our own 03_tune_and_finalize.py in this same outputs
    # folder in the previous pipeline step, not sourced externally.
    pipe = joblib.load(OUT_DIR / "final_model_pipeline.joblib")
    train_df = pd.read_csv(OUT_DIR / "train_split.csv")
    test_df = pd.read_csv(OUT_DIR / "test_split.csv")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET].values

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    X_train_t = prep.transform(X_train)
    X_test_t = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    X_train_t_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_t_df = pd.DataFrame(X_test_t, columns=feature_names)

    # background sample for the explainer (speed)
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train_t_df), size=min(200, len(X_train_t_df)), replace=False)
    background = X_train_t_df.iloc[bg_idx]

    print(f"Final model class: {type(clf).__name__}")
    # Evaluate SHAP on a capped sample of the test set for speed; the driver
    # ranking/direction is a population-level summary and doesn't need every row.
    rng2 = np.random.RandomState(RANDOM_STATE)
    eval_idx = rng2.choice(len(X_test_t_df), size=min(400, len(X_test_t_df)), replace=False)
    X_eval = X_test_t_df.iloc[eval_idx]

    try:
        # Pass the model object itself (not a bound method) so SHAP can
        # auto-select the fast, exact Tree or Linear explainer instead of
        # falling back to the much slower generic Permutation explainer.
        explainer = shap.Explainer(clf, background)
        shap_out = explainer(X_eval)
    except Exception as e:
        print(f"Direct model explainer failed ({e}); falling back to predict_proba-based explainer.")
        explainer = shap.Explainer(clf.predict_proba, background, algorithm="auto")
        shap_out = explainer(X_eval)

    # for binary classifiers shap may return values for both classes; keep class-1
    values = shap_out.values
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = np.abs(values).mean(axis=0)

    # direction: correlation between each feature's value and its shap value
    directions = []
    for i, fname in enumerate(feature_names):
        fvals = X_eval.iloc[:, i].values
        svals = values[:, i]
        if np.std(fvals) > 0 and np.std(svals) > 0:
            corr = np.corrcoef(fvals, svals)[0, 1]
        else:
            corr = np.nan
        directions.append(corr)

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "direction_corr(+feature -> +risk if positive)": directions,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(OUT_DIR / "feature_importance_shap.csv", index=False)
    print("\nTop 15 features by mean |SHAP|:")
    print(shap_importance.head(15).to_string(index=False))

    # SHAP summary bar plot
    plt.figure(figsize=(8, 6))
    order = np.argsort(-mean_abs_shap)[:15]
    plt.barh(np.array(feature_names)[order][::-1], mean_abs_shap[order][::-1])
    plt.xlabel("mean |SHAP value|  (impact on predicted churn probability)")
    plt.title("Top 15 churn drivers (SHAP, held-out test set)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=130)
    print(f"\nSaved SHAP summary plot to {OUT_DIR / 'shap_summary.png'}")

    # ---- permutation importance (model-agnostic cross-check) ----
    def pr_auc_scorer(estimator, X, y):
        return average_precision_score(y, estimator.predict_proba(X)[:, 1])

    perm = permutation_importance(
        pipe, X_test, y_test, scoring=pr_auc_scorer,
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "perm_importance_mean_drop_in_pr_auc": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean_drop_in_pr_auc", ascending=False)
    perm_df.to_csv(OUT_DIR / "feature_importance_permutation.csv", index=False)
    print("\nPermutation importance (raw, pre-encoding features), drop in test PR-AUC when shuffled:")
    print(perm_df.to_string(index=False))


if __name__ == "__main__":
    main()
