"""
Stage 9 - Evaluation + sign-off (ds-evaluation).

Preconditions checked: verify_test_untouched()==True; threshold is a parameter-free rank rule
(call the top 10% by score) fixed at Stage 0, not tuned on test; pipeline/hyperparameters fixed
by Stage 8 before this script ever looks at test.

Final pipeline: LogisticRegression(C=0.0192, class_weight="balanced", max_iter=3000) on 8 raw
features (tenure_months, monthly_charges, total_charges, support_calls_90d, avg_daily_usage_mb,
age, contract_type, payment_method), median imputation, StandardScaler, one-hot encoding.
Refit on train+holdout (never test), then scored on test exactly once below.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-evaluation/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-method-comparison/scripts")
from cv_folds import load_split, verify_test_untouched
from final_eval import (bootstrap_ci, capacity_threshold, confusion_at, calibration_table,
                         slice_table, shuffled_label_check, interval_coverage)
from screen import topk_scorer

df = pd.read_csv(OUT + "/cleaned_data.csv")
s = load_split(OUT + "/split/split.csv", ids=df["customer_id"])
print("verify_test_untouched:", verify_test_untouched(OUT + "/split"))

FEATS = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
         "avg_daily_usage_mb", "age", "contract_type", "payment_method"]
num_feats = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
cat_feats = ["contract_type", "payment_method"]

train = df[s.train_mask].reset_index(drop=True)
holdout = df[s.holdout_mask].reset_index(drop=True)
test = df[s.test_mask].reset_index(drop=True)
train_ho = pd.concat([train, holdout], ignore_index=True)

def make_pick_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.019150292626953554, max_iter=3000, class_weight="balanced"))])

# ---- Step 2: threshold, chosen on holdout (train-only fit), BEFORE touching test ----
pipe_trainfit = make_pick_pipe().fit(train[FEATS], train["churned"])
ho_scores = pipe_trainfit.predict_proba(holdout[FEATS])[:, 1]
ho_thresh = capacity_threshold(ho_scores, 0.10)
ho_conf = confusion_at(holdout["churned"].to_numpy(), ho_scores, ho_thresh)
print("\n=== Threshold chosen on HOLDOUT (pre-registered, before test is touched) ===")
print(f"capacity threshold (top 10% by score) = {ho_thresh:.4f}")
print(f"holdout precision@10%={ho_conf['precision']:.4f} recall@10%={ho_conf['recall']:.4f} flagged_share={ho_conf['flagged_share']:.4f}")

# ---- Step 1 cont'd / final refit: train + holdout, never test ----
pick = make_pick_pipe().fit(train_ho[FEATS], train_ho["churned"])

# baselines, refit the same way, scored on the same test rows
def risk_rule_score(Xdf):
    return (3 * (Xdf["contract_type"] == "month-to-month").astype(int)
            + 1 * Xdf["support_calls_90d"].fillna(0)
            + 0.5 * (Xdf["payment_method"] == "Electronic check").astype(int))

class BusinessRule(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        sc = risk_rule_score(X).to_numpy(dtype=float)
        sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        return np.column_stack([1 - sc, sc])

dummy = DummyClassifier(strategy="prior").fit(train_ho[FEATS], train_ho["churned"])
rule = BusinessRule().fit(train_ho[FEATS], train_ho["churned"])
tree_pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_feats),
                               ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats)])
shallow_tree = Pipeline([("pre", tree_pre), ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0))]).fit(train_ho[FEATS], train_ho["churned"])

y_test = test["churned"].to_numpy()
scores = {
    "trivial_dummy": dummy.predict_proba(test[FEATS])[:, 1],
    "business_rule": rule.predict_proba(test[FEATS])[:, 1],
    "shallow_tree_d3": shallow_tree.predict_proba(test[FEATS])[:, 1],
    "final_pipeline_logreg": pick.predict_proba(test[FEATS])[:, 1],
}

# ---- Step 5: score the test set, ONCE, with bootstrap CIs, all models on the same rows ----
print("\n=== TEST SET (scored once) ===")
rows = []
for name, sc in scores.items():
    ap, ap_lo, ap_hi = bootstrap_ci(y_test, sc, average_precision_score)
    auc, auc_lo, auc_hi = bootstrap_ci(y_test, sc, roc_auc_score)
    thr = capacity_threshold(sc, 0.10)
    conf = confusion_at(y_test, sc, thr)
    prec10, prec_lo, prec_hi = bootstrap_ci(y_test, sc, lambda y, s: confusion_at(y, s, capacity_threshold(s, 0.10))["precision"])
    rec10, rec_lo, rec_hi = bootstrap_ci(y_test, sc, lambda y, s: confusion_at(y, s, capacity_threshold(s, 0.10))["recall"])
    rows.append({"model": name, "PR_AUC": round(ap, 4), "PR_AUC_lo": round(ap_lo, 4), "PR_AUC_hi": round(ap_hi, 4),
                 "ROC_AUC": round(auc, 4), "ROC_AUC_lo": round(auc_lo, 4), "ROC_AUC_hi": round(auc_hi, 4),
                 "precision_at_10": round(prec10, 4), "precision_at_10_lo": round(prec_lo, 4), "precision_at_10_hi": round(prec_hi, 4),
                 "recall_at_10": round(rec10, 4), "recall_at_10_lo": round(rec_lo, 4), "recall_at_10_hi": round(rec_hi, 4),
                 "n_flagged": conf["tp"] + conf["fp"], "n_true_positive_caught": conf["tp"]})
    print(f"{name:24s} PR-AUC={ap:.4f} ({ap_lo:.4f}-{ap_hi:.4f})  ROC-AUC={auc:.4f} ({auc_lo:.4f}-{auc_hi:.4f})  "
          f"P@10%={prec10:.4f} ({prec_lo:.4f}-{prec_hi:.4f})  R@10%={rec10:.4f} ({rec_lo:.4f}-{rec_hi:.4f})")
test_table = pd.DataFrame(rows)
test_table.to_csv(OUT + "/test_results.csv", index=False)

# "honest ceiling" sanity: compare against the CV mean / holdout to check no leak survived
print(f"\nsanity: CV mean PR-AUC (Stage 8) = 0.2174, holdout PR-AUC (Stage 8) = 0.2089, "
      f"test PR-AUC = {rows[-1]['PR_AUC']:.4f} -> all three should agree within noise")

# ---- Step 3: calibration (diagnostic; scores are used for RANKING, not as raw probabilities) ----
cal_tab, ece, brier = calibration_table(y_test, scores["final_pipeline_logreg"])
print(f"\n=== Calibration (diagnostic) === ECE={ece:.4f} Brier={brier:.4f}")
print(cal_tab)
cal_tab.to_csv(OUT + "/calibration_table.csv")

# ---- Step 6: slices ----
slice_cols = test[["contract_type", "payment_method", "region"]].copy()
slice_cols["tenure_band"] = pd.cut(test["tenure_months"], bins=[0, 8, 24, 48, 72], labels=["0-8mo", "9-24mo", "25-48mo", "49-72mo"])
slice_cols["had_missing_last_invoice"] = test["last_invoice_amount"].isna().map({True: "missing", False: "present"})
stab = slice_table(slice_cols, y_test, scores["final_pipeline_logreg"], average_precision_score, min_n=30)
stab.to_csv(OUT + "/slice_table.csv", index=False)
print("\n=== Slice table (PR-AUC per segment, min_n=30) ===")
print(stab.to_string())

# ---- Step 7: error analysis ----
test_err = test.copy()
test_err["score"] = scores["final_pipeline_logreg"]
worst_fn = test_err[test_err["churned"] == 1].sort_values("score").head(10)
worst_fp = test_err[test_err["churned"] == 0].sort_values("score", ascending=False).head(10)
worst_fn.to_csv(OUT + "/error_analysis_worst_false_negatives.csv", index=False)
worst_fp.to_csv(OUT + "/error_analysis_worst_false_positives.csv", index=False)
print("\nworst 10 missed churners (lowest score, churned=1):")
print(worst_fn[["customer_id", "score", "tenure_months", "contract_type", "support_calls_90d", "monthly_charges"]].to_string())
print("\nworst 10 false alarms (highest score, churned=0):")
print(worst_fp[["customer_id", "score", "tenure_months", "contract_type", "support_calls_90d", "monthly_charges"]].to_string())

# ---- Step 8: sanity tests ----
real_ap, fake_ap = shuffled_label_check(make_pick_pipe(), train[FEATS], train["churned"], s.folds,
                                         lambda est, X, y: average_precision_score(y, est.predict_proba(X)[:, 1]))
print(f"\nsanity: shuffled-label check (PR-AUC) real={real_ap:.4f} shuffled={fake_ap:.4f} "
      f"(base rate={train['churned'].mean():.4f}) - shuffled should fall to ~base rate")

dup_row = test[FEATS].iloc[[0, 0]]
dup_scores = pick.predict_proba(dup_row)[:, 1]
print(f"sanity: duplicate row -> identical prediction: {dup_scores[0]:.6f} vs {dup_scores[1]:.6f} (match={np.isclose(dup_scores[0], dup_scores[1])})")

raw_messy = pd.DataFrame([{
    "tenure_months": 24, "monthly_charges": 70.0, "total_charges": -999,  # sentinel, never cleaned
    "support_calls_90d": 1, "avg_daily_usage_mb": 15.0, "age": 250,  # impossible, never cleaned
    "contract_type": "TWO YEAR", "payment_method": "Mailed check",  # odd spelling
}])
try:
    p_messy = pick.predict_proba(raw_messy)[:, 1]
    print(f"sanity: raw/messy row (sentinel total_charges, impossible age, odd casing) through the pipeline -> "
          f"p={p_messy[0]:.4f} (NOTE: this pipeline does not re-apply Stage-1 stateless cleaning, "
          f"so -999/age=250 pass straight to the imputer as if genuine numeric values - see Open risk below)")
except Exception as e:
    print(f"sanity: raw/messy row FAILED: {e}")

all_missing = pd.DataFrame([{c: (np.nan if c in num_feats else None) for c in FEATS}])
p_allmiss = pick.predict_proba(all_missing)[:, 1]
print(f"sanity: all-features-missing row -> p={p_allmiss[0]:.4f} (ok={not np.isnan(p_allmiss[0])})")

# ---- Step 9: explanations ----
coefs = pick.named_steps["clf"].coef_[0]
feat_names = pick.named_steps["pre"].get_feature_names_out()
coef_tab = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
coef_tab.to_csv(OUT + "/model_coefficients.csv", index=False)
print("\n=== Logistic regression coefficients (standardized features -> directly comparable) ===")
print(coef_tab.to_string(index=False))

perm = permutation_importance(pick, test[FEATS], y_test, scoring="average_precision", n_repeats=20, random_state=0)
perm_tab = pd.DataFrame({"feature": FEATS, "importance_mean": perm.importances_mean, "importance_sd": perm.importances_std}).sort_values("importance_mean", ascending=False)
perm_tab.to_csv(OUT + "/permutation_importance.csv", index=False)
print("\n=== Permutation importance (test set, PR-AUC drop) ===")
print(perm_tab.to_string(index=False))

with open(OUT + "/search/result.json") as f:
    result = json.load(f)
result["test_evaluation"] = {"table": rows, "holdout_precision_at_10": ho_conf["precision"],
                              "holdout_recall_at_10": ho_conf["recall"], "capacity_threshold_holdout": ho_thresh,
                              "ece": ece, "brier": brier,
                              "shuffled_label_check_real_prauc": real_ap, "shuffled_label_check_shuffled_prauc": fake_ap}
with open(OUT + "/search/result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("\nwrote test_results.csv, calibration_table.csv, slice_table.csv, error_analysis_*.csv, "
      "model_coefficients.csv, permutation_importance.csv; updated search/result.json")
