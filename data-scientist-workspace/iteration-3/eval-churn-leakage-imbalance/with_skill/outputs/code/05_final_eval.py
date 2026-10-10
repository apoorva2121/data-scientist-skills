"""Stage 9: evaluation + sign-off. Test set scored exactly once, here.
Final pipeline = Stage 8 winner (T22: linear/logistic, median_indicator imputation, no outlier
treatment, onehot encoding, tuned C=0.0101, class_weight=balanced), refit on train+holdout (all
non-test rows), then scored once on the frozen test set.
"""
import sys, json
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-evaluation\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
import numpy as np, pandas as pd
from cv_folds import load_split, verify_test_untouched
from screen import topk_scorer
from final_eval import bootstrap_ci, calibration_table, confusion_at, capacity_threshold, slice_table, shuffled_label_check
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.dummy import DummyClassifier

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-3\eval-churn-leakage-imbalance\with_skill\outputs"
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
       "avg_daily_usage_mb", "age", "last_invoice_amount"]
CAT = ["contract_type", "payment_method", "region"]

df = pd.read_csv(f"{OUT}/cleaned.csv")
s = load_split(f"{OUT}/split/split.csv", ids=df["customer_id"])
assert verify_test_untouched(f"{OUT}/split") is True, "test set was touched before Stage 9"

def make_final_pipe():
    num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([("num", num_pipe, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    model = LogisticRegression(C=0.010074528943967254, max_iter=3000, class_weight="balanced", random_state=0)
    return Pipeline([("pre", pre), ("model", model)])

trho = df[s.train_mask | s.holdout_mask].reset_index(drop=True)  # train+holdout for final fit
test = df[s.test_mask].reset_index(drop=True)
Xtr, ytr = trho[NUM + CAT], trho["churned"].to_numpy()
Xte, yte = test[NUM + CAT], test["churned"].to_numpy()

pipe = make_final_pipe().fit(Xtr, ytr)
p_test = pipe.predict_proba(Xte)[:, 1]

scorer = topk_scorer(0.10, "precision")
recall_scorer = topk_scorer(0.10, "recall")

def prec10(y, s_):
    thr = capacity_threshold(s_, 0.10)
    return confusion_at(y, s_, thr)["precision"]

def rec10(y, s_):
    thr = capacity_threshold(s_, 0.10)
    return confusion_at(y, s_, thr)["recall"]

prec_point, prec_lo, prec_hi = bootstrap_ci(yte, p_test, prec10, n_boot=2000, seed=0)
rec_point, rec_lo, rec_hi = bootstrap_ci(yte, p_test, rec10, n_boot=2000, seed=0)
ap_point, ap_lo, ap_hi = bootstrap_ci(yte, p_test, average_precision_score, n_boot=2000, seed=0)
auc_point, auc_lo, auc_hi = bootstrap_ci(yte, p_test, roc_auc_score, n_boot=2000, seed=0)

base_rate = yte.mean()
lift = prec_point / base_rate

print(f"TEST precision@10%: {prec_point:.4f} [{prec_lo:.4f}, {prec_hi:.4f}]  (base rate {base_rate:.4f}, lift {lift:.2f}x)")
print(f"TEST recall@10%   : {rec_point:.4f} [{rec_lo:.4f}, {rec_hi:.4f}]")
print(f"TEST avg precision (PR-AUC): {ap_point:.4f} [{ap_lo:.4f}, {ap_hi:.4f}]")
print(f"TEST ROC-AUC      : {auc_point:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

# calibration
cal_table, ece, brier = calibration_table(yte, p_test, n_bins=10)
print(f"ECE={ece:.4f} Brier={brier:.4f}")
cal_table.to_csv(f"{OUT}/test_calibration.csv", index=False)

# threshold / confusion at capacity
thr = capacity_threshold(p_test, 0.10)
conf = confusion_at(yte, p_test, thr)
print("confusion@10%:", conf, "threshold(score)=", thr)

# slices: region, contract_type, tenure bucket
test_sl = test.copy()
test_sl["tenure_bucket"] = pd.cut(test_sl["tenure_months"], [0, 12, 36, 100], labels=["0-12mo", "13-36mo", "37mo+"])
slices = test_sl[["region", "contract_type", "tenure_bucket"]]
sl_table = slice_table(slices, yte, p_test, prec10, min_n=30, n_boot=300)
sl_table.to_csv(f"{OUT}/test_slices.csv", index=False)
print(sl_table.to_string(index=False))

# sanity: shuffled label check on CV folds (train part) - real vs shuffled labels should diverge
tr_only = df[s.train_mask].reset_index(drop=True)
Xtr2, ytr2 = tr_only[NUM + CAT], tr_only["churned"].to_numpy()
real, shuf = shuffled_label_check(make_final_pipe(), Xtr2, ytr2, s.folds, scorer, seed=0)
print(f"shuffled-label check: real={real:.4f} shuffled={shuf:.4f} (shuffled should collapse toward trivial ~{ytr2.mean():.3f})")

# feature importance: standardized logistic regression coefficients
ohe = pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(CAT))
num_imputer = pipe.named_steps["pre"].named_transformers_["num"].named_steps["simpleimputer"]
ind_names = [f"{NUM[i]}_missing" for i in num_imputer.indicator_.features_] if num_imputer.indicator_ is not None else []
feat_names = NUM + ind_names + cat_names
coefs = pipe.named_steps["model"].coef_[0]
imp = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
imp.to_csv(f"{OUT}/feature_importance.csv", index=False)
print(imp.head(15).to_string(index=False))

result = {
    "test_precision_at_10pct": [prec_point, prec_lo, prec_hi],
    "test_recall_at_10pct": [rec_point, rec_lo, rec_hi],
    "test_average_precision": [ap_point, ap_lo, ap_hi],
    "test_roc_auc": [auc_point, auc_lo, auc_hi],
    "base_rate_test": float(base_rate),
    "lift_at_10pct": float(lift),
    "ece": float(ece), "brier": float(brier),
    "capacity_threshold_score": float(thr),
    "confusion_at_10pct": conf,
    "shuffled_label_check": {"real": real, "shuffled": shuf},
    "n_test": int(len(test)), "n_test_positive": int(yte.sum()),
}
with open(f"{OUT}/test_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("\nwritten test_results.json")

import joblib
joblib.dump(pipe, f"{OUT}/final_pipeline.joblib")
print("written final_pipeline.joblib")
