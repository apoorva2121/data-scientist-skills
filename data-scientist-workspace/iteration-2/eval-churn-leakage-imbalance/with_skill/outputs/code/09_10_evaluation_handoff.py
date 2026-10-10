"""
Stage 9: evaluation + sign-off (test set scored once)
Stage 10: handoff (save pipeline, model card, monitoring plan)
"""
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance
import joblib

RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-churn-leakage-imbalance\with_skill\outputs"
SEED = 42
RNG = np.random.RandomState(SEED)

def _canon_contract(s):
    if pd.isna(s):
        return s
    t = str(s).strip().lower()
    return {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}.get(t, s)

def clean_raw(df):
    """Stateless Stage-1 cleaning. Reused as the first step of the production pipeline."""
    d = df.copy()
    for c in ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
              "avg_daily_usage_mb", "age", "last_invoice_amount"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "churned" in d.columns:
        d["churned"] = pd.to_numeric(d["churned"], errors="coerce")
    d.loc[d["total_charges"] == -999, "total_charges"] = np.nan
    d.loc[(d["age"] <= 0) | (d["age"] >= 120), "age"] = np.nan
    d["contract_type"] = d["contract_type"].apply(_canon_contract)
    d = d.drop_duplicates().reset_index(drop=True)
    return d

def add_engineered(X):
    X = X.copy()
    X["charge_gap"] = X["monthly_charges"] * X["tenure_months"] - X["total_charges"]
    X["is_new_customer"] = (X["tenure_months"] <= 3).astype(int)
    return X

NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
       "avg_daily_usage_mb", "age", "last_invoice_amount"]
CAT = ["contract_type", "payment_method", "region"]
NUM_FE = NUM + ["charge_gap", "is_new_customer"]

df_raw = pd.read_csv(RAW, dtype=str)
df = clean_raw(df_raw).drop(columns=["churn_reason", "account_status_code"])
split = pd.read_csv(f"{OUT}/split.csv")
df = df.merge(split, on="customer_id")

train = df[df["part"] == "train"].reset_index(drop=True)
holdout = df[df["part"] == "holdout"].reset_index(drop=True)
test = df[df["part"] == "test"].reset_index(drop=True)

with open(f"{OUT}/split_spec.json") as f:
    spec = json.load(f)
test_ids_sha_now = hashlib.sha256(",".join(sorted(test["customer_id"])).encode()).hexdigest()
assert test_ids_sha_now == spec["test_ids_sha256"], "TEST SET WAS MODIFIED - ABORT"
print("verify_test_untouched: True (sha256 match)")

y_train = train["churned"].astype(int)
y_holdout = holdout["churned"].astype(int)
y_test = test["churned"].astype(int)

Xtr = add_engineered(train[NUM + CAT])
Xho = add_engineered(holdout[NUM + CAT])
Xte = add_engineered(test[NUM + CAT])
Xtrho = pd.concat([Xtr, Xho], ignore_index=True)
ytrho = pd.concat([y_train, y_holdout], ignore_index=True)

def prep_linear():
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUM_FE), ("cat", cat_pipe, CAT)])

# ---------------- Stage 9 Step 1: refit winner on train+holdout ----------------
final_pipe = Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01))])
final_pipe.fit(Xtrho, ytrho)

# ---------------- Step 2: threshold, chosen on holdout (capacity = 10%) ----------------
proba_ho = Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01))]).fit(Xtr, y_train).predict_proba(Xho)[:, 1]
k = max(1, int(round(len(y_holdout) * 0.10)))
thr = np.sort(proba_ho)[::-1][k - 1]
print(f"Capacity threshold (from holdout, top 10%): {thr:.4f}")

# ---------------- Step 5: score test set ONCE ----------------
proba_test = final_pipe.predict_proba(Xte)[:, 1]

def precision_at_k(y_true, scores, k_frac=0.10):
    n = len(y_true); kk = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    return np.asarray(y_true)[order[:kk]].mean(), kk

def recall_at_k(y_true, scores, k_frac=0.10):
    n = len(y_true); kk = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    top = np.asarray(y_true)[order[:kk]]
    return top.sum() / max(1, np.asarray(y_true).sum())

def bootstrap_ci(y_true, scores, fn, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            v = fn(y_true[idx], scores[idx])
            if isinstance(v, tuple):
                v = v[0]
            vals.append(v)
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

p10_point, k10 = precision_at_k(y_test.values, proba_test)
p10_mean, p10_lo, p10_hi = bootstrap_ci(y_test.values, proba_test, lambda yt, sc: precision_at_k(yt, sc)[0])
auc_mean, auc_lo, auc_hi = bootstrap_ci(y_test.values, proba_test, roc_auc_score)
prauc_mean, prauc_lo, prauc_hi = bootstrap_ci(y_test.values, proba_test, average_precision_score)
rec10 = recall_at_k(y_test.values, proba_test)
brier = brier_score_loss(y_test, proba_test)

test_results = {
    "precision_at_10pct": {"point": p10_point, "ci95": [p10_lo, p10_hi], "n_called": int(k10)},
    "recall_captured_at_10pct": rec10,
    "roc_auc": {"point": float(roc_auc_score(y_test, proba_test)), "ci95": [auc_lo, auc_hi]},
    "pr_auc": {"point": float(average_precision_score(y_test, proba_test)), "ci95": [prauc_lo, prauc_hi]},
    "brier_score": float(brier),
    "base_rate_test": float(y_test.mean()),
    "lift_over_base_rate": float(p10_point / y_test.mean()),
}
with open(f"{OUT}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)
print(json.dumps(test_results, indent=2))

# same metrics for baselines on the SAME test rows, for the results table
rule_score_test = test["support_calls_90d"].values + (test["contract_type"] == "Month-to-month").astype(int).values * 0.5
p10_rule, _ = precision_at_k(y_test.values, rule_score_test)
trivial_p10 = y_test.mean()

# ---------------- Step 3: calibration table ----------------
cal_rows = []
dec = pd.qcut(proba_test, 10, duplicates="drop")
cal_df = pd.DataFrame({"proba": proba_test, "y": y_test.values, "decile": dec})
for d, g in cal_df.groupby("decile", observed=True):
    cal_rows.append({"decile": str(d), "n": len(g), "mean_predicted": g.proba.mean(), "observed_rate": g.y.mean()})
pd.DataFrame(cal_rows).to_csv(f"{OUT}/calibration_table.csv", index=False)

# ---------------- Step 6: slices ----------------
slice_rows = []
test_aug = test.copy()
test_aug["proba"] = proba_test
test_aug["tenure_band"] = pd.cut(test_aug["tenure_months"], [0, 12, 36, 100], labels=["0-12mo", "13-36mo", "37mo+"])
for col in ["contract_type", "tenure_band"]:
    for level, g in test_aug.groupby(col, observed=True):
        if len(g) < 30:
            continue
        p10_s, ks = precision_at_k(g["churned"].values, g["proba"].values, 0.10)
        slice_rows.append({"slice_col": col, "level": str(level), "n": len(g),
                            "base_rate": g["churned"].mean(), "precision_at_10pct": p10_s,
                            "roc_auc": roc_auc_score(g["churned"], g["proba"]) if g["churned"].nunique() > 1 else None})
slice_df = pd.DataFrame(slice_rows)
slice_df.to_csv(f"{OUT}/slice_table.csv", index=False)
print(slice_df.to_string(index=False))

# ---------------- Step 8: sanity tests ----------------
sanity = {}
# shuffled label check on train folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
y_shuf = y_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
proba_shuf = cross_val_predict(Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01))]),
                                Xtr, y_shuf, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
sanity["shuffled_label_pr_auc"] = float(average_precision_score(y_shuf, proba_shuf))
sanity["shuffled_label_expected_near"] = float(y_train.mean())
# duplicate row -> identical prediction
row = Xte.iloc[[0]]
p1 = final_pipe.predict_proba(row)[0, 1]
p2 = final_pipe.predict_proba(pd.concat([row, row]))[:, 1]
sanity["duplicate_row_identical"] = bool(np.isclose(p2[0], p2[1]) and np.isclose(p1, p2[0]))
# raw uncleaned row end to end
raw_sample = df_raw.sample(3, random_state=SEED)
raw_clean = clean_raw(raw_sample)
raw_fe = add_engineered(raw_clean[NUM + CAT])
try:
    raw_scores = final_pipe.predict_proba(raw_fe)[:, 1]
    sanity["raw_rows_scored_ok"] = True
    sanity["raw_rows_scores"] = raw_scores.tolist()
except Exception as e:
    sanity["raw_rows_scored_ok"] = False
    sanity["raw_rows_error"] = str(e)
# feed one row twice -> identical
sanity["repeat_scoring_identical"] = bool(np.allclose(final_pipe.predict_proba(row)[:, 1], final_pipe.predict_proba(row)[:, 1]))
with open(f"{OUT}/sanity_checks.json", "w") as f:
    json.dump(sanity, f, indent=2)
print(json.dumps(sanity, indent=2))

# ---------------- Step 9: explanations ----------------
feat_names = final_pipe.named_steps["prep"].get_feature_names_out()
coefs = final_pipe.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
coef_df.to_csv(f"{OUT}/coefficients.csv", index=False)

perm = permutation_importance(final_pipe, Xte, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame({"feature": Xte.columns, "importance_mean": perm.importances_mean, "importance_sd": perm.importances_std}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(f"{OUT}/permutation_importance.csv", index=False)
print(coef_df.head(10).to_string(index=False))
print(perm_df.to_string(index=False))

# ---------------- Stage 10: save pipeline ----------------
import sklearn, lightgbm, xgboost
joblib.dump({"pipeline": final_pipe, "clean_raw": None, "threshold": float(thr),
             "feature_cols_raw": NUM + CAT, "sklearn_version": sklearn.__version__}, f"{OUT}/final_model.joblib")

print("Stage 9-10 artifacts written.")
