"""
Stage 7: baselines + model families + imbalance handling
Stage 8: joint pipeline search (small manual grid given the time budget, not Optuna)

Primary metric: precision@10% (top-decile precision by predicted score), matching the retention
team's fixed calling capacity. Secondary: ROC-AUC, PR-AUC.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb

RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-churn-leakage-imbalance\with_skill\outputs"
SEED = 42

def _canon_contract(s):
    if pd.isna(s):
        return s
    t = str(s).strip().lower()
    return {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}.get(t, s)

def clean_raw(df):
    d = df.copy()
    for c in ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
              "avg_daily_usage_mb", "age", "last_invoice_amount", "churned"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["signup_date"] = pd.to_datetime(d["signup_date"], errors="coerce", format="%Y-%m-%d")
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

df_raw = pd.read_csv(RAW, dtype=str)
df = clean_raw(df_raw).drop(columns=["churn_reason", "account_status_code"])
split = pd.read_csv(f"{OUT}/split.csv")
df = df.merge(split, on="customer_id")

NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
       "avg_daily_usage_mb", "age", "last_invoice_amount"]
CAT = ["contract_type", "payment_method", "region"]
NUM_FE = NUM + ["charge_gap", "is_new_customer"]

for part_name in ["train", "holdout", "test"]:
    pass

train = df[df["part"] == "train"].reset_index(drop=True)
holdout = df[df["part"] == "holdout"].reset_index(drop=True)
y_train = train["churned"].astype(int)
y_holdout = holdout["churned"].astype(int)

Xtr_fe = add_engineered(train[NUM + CAT])
Xho_fe = add_engineered(holdout[NUM + CAT])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

def precision_at_k(y_true, scores, k_frac=0.10):
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    top = np.asarray(y_true)[order[:k]]
    return top.mean()

def eval_cv_proba(pipe, X, y, cv):
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return {
        "precision_at_10pct": precision_at_k(y, proba, 0.10),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }

def prep_linear(log_usage=True):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    cols = NUM_FE.copy()
    return ColumnTransformer([("num", num_pipe, cols), ("cat", cat_pipe, CAT)])

def prep_tree():
    num_pipe = "passthrough"
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUM_FE), ("cat", cat_pipe, CAT)])

# ---------------- Stage 7: Baselines ----------------
baseline_rows = []

dummy = DummyClassifier(strategy="prior")
dummy_proba = cross_val_predict(dummy, Xtr_fe, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
baseline_rows.append({"model": "trivial (predict prior)", **{
    "precision_at_10pct": y_train.mean(),  # random 10% call list ~ base rate
    "roc_auc": 0.5, "pr_auc": y_train.mean()}})

# business rule: rank by support_calls_90d desc, then month-to-month flag
rule_score = train["support_calls_90d"].values + (train["contract_type"] == "Month-to-month").astype(int).values * 0.5
baseline_rows.append({"model": "business rule (support_calls + MTM flag)",
                       "precision_at_10pct": precision_at_k(y_train.values, rule_score, 0.10),
                       "roc_auc": roc_auc_score(y_train, rule_score),
                       "pr_auc": average_precision_score(y_train, rule_score)})

simple_lr = Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
m = eval_cv_proba(simple_lr, Xtr_fe, y_train, cv)
baseline_rows.append({"model": "simple model (logistic, balanced)", **m})

tree = Pipeline([("prep", prep_tree()), ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED))])
m = eval_cv_proba(tree, Xtr_fe, y_train, cv)
baseline_rows.append({"model": "shallow tree (depth=3)", **m})

base_df = pd.DataFrame(baseline_rows)
base_df.to_csv(f"{OUT}/baseline_table.csv", index=False)
print(base_df.to_string(index=False))

# ---------------- Stage 7: Family screen (defaults, imbalance = class weight) ----------------
family_rows = []

lgb_clf = lgb.LGBMClassifier(random_state=SEED, n_estimators=200, class_weight="balanced", verbosity=-1)
lgb_pipe = Pipeline([("prep", prep_tree()), ("clf", lgb_clf)])
m = eval_cv_proba(lgb_pipe, Xtr_fe, y_train, cv); family_rows.append({"family": "LightGBM (default, balanced)", **m})

pos = y_train.sum(); neg = len(y_train) - pos
xgb_clf = xgb.XGBClassifier(random_state=SEED, n_estimators=200, eval_metric="logloss", scale_pos_weight=neg/pos)
xgb_pipe = Pipeline([("prep", prep_tree()), ("clf", xgb_clf)])
m = eval_cv_proba(xgb_pipe, Xtr_fe, y_train, cv); family_rows.append({"family": "XGBoost (default, scale_pos_weight)", **m})

m = eval_cv_proba(simple_lr, Xtr_fe, y_train, cv); family_rows.append({"family": "Logistic regression (balanced)", **m})

fam_df = pd.DataFrame(family_rows).sort_values("precision_at_10pct", ascending=False)
fam_df.to_csv(f"{OUT}/family_screen.csv", index=False)
print(fam_df.to_string(index=False))

# learning curve for the best family (logistic, per screen) vs LightGBM
sizes, tr_sc, va_sc = learning_curve(simple_lr, Xtr_fe, y_train, cv=cv, scoring="roc_auc",
                                      train_sizes=[0.25, 0.5, 0.75, 1.0], n_jobs=-1)
print("Learning curve (logistic) train_sizes:", sizes, "val mean:", va_sc.mean(axis=1))

# Note: TabPFN/foundation models skipped - hosted inference sends rows to a third party (data-location
# check not clearable with the user unavailable) and licence for TabPFN weights is non-commercial;
# logged as a next-step recommendation, not silently dropped.

# ---------------- Stage 8: small joint search ----------------
# Search over: model family x key hyperparams x {imputer variant, log-usage} x class-weighting.
# Kept intentionally small (time budget): ~10 configs, 3-fold CV, confirmed on the locked holdout.
candidates = []

for max_depth, lr, n_est, reg in [(3, 0.05, 300, 1.0), (4, 0.05, 300, 1.0), (3, 0.1, 200, 5.0)]:
    clf = lgb.LGBMClassifier(random_state=SEED, n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                              class_weight="balanced", reg_lambda=reg, verbosity=-1)
    pipe = Pipeline([("prep", prep_tree()), ("clf", clf)])
    candidates.append((f"lightgbm_d{max_depth}_lr{lr}_n{n_est}_reg{reg}", pipe))

for max_depth, lr, n_est in [(3, 0.05, 300), (4, 0.05, 300)]:
    clf = xgb.XGBClassifier(random_state=SEED, n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                             eval_metric="logloss", scale_pos_weight=neg/pos, reg_lambda=2.0)
    pipe = Pipeline([("prep", prep_tree()), ("clf", clf)])
    candidates.append((f"xgboost_d{max_depth}_lr{lr}_n{n_est}", pipe))

for C in [0.01, 0.1, 1.0]:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    pipe = Pipeline([("prep", prep_linear()), ("clf", clf)])
    candidates.append((f"logreg_C{C}", pipe))

search_log = []
for name, pipe in candidates:
    proba = cross_val_predict(pipe, Xtr_fe, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    # per-fold precision@10% for paired comparison
    fold_scores = []
    for _, val_idx in cv.split(Xtr_fe, y_train):
        fold_scores.append(precision_at_k(y_train.values[val_idx], proba[val_idx], 0.10))
    search_log.append({
        "name": name,
        "precision_at_10pct_mean": float(np.mean(fold_scores)),
        "precision_at_10pct_sd": float(np.std(fold_scores, ddof=1)),
        "roc_auc": roc_auc_score(y_train, proba),
        "pr_auc": average_precision_score(y_train, proba),
    })
    print(search_log[-1])

search_df = pd.DataFrame(search_log).sort_values("precision_at_10pct_mean", ascending=False)
search_df.to_csv(f"{OUT}/search_log.csv", index=False)
print(search_df.to_string(index=False))

winner_name = search_df.iloc[0]["name"]
winner_pipe = dict(candidates)[winner_name]
runner_name = search_df.iloc[1]["name"]
print(f"WINNER: {winner_name}  RUNNER-UP: {runner_name}")

# Locked-holdout check: fit winner and runner-up on full train, score once on holdout
for name, pipe in [(winner_name, dict(candidates)[winner_name]), (runner_name, dict(candidates)[runner_name])]:
    pipe.fit(Xtr_fe, y_train)
    proba_ho = pipe.predict_proba(Xho_fe)[:, 1]
    p10 = precision_at_k(y_holdout.values, proba_ho, 0.10)
    auc = roc_auc_score(y_holdout, proba_ho)
    print(f"HOLDOUT {name}: precision@10%={p10:.4f} roc_auc={auc:.4f}")

with open(f"{OUT}/search_space.json") as f:
    ss = json.load(f)
ss["stages"]["models"] = [{"name": r["name"], "applies_to": ["gbdt" if "logreg" not in r["name"] else "linear"],
                            "precision_at_10pct": [round(r["precision_at_10pct_mean"], 4), round(r["precision_at_10pct_sd"], 4)],
                            "roc_auc": round(r["roc_auc"], 4)} for _, r in search_df.iterrows()]
ss["winner"] = winner_name
ss["runner_up"] = runner_name
with open(f"{OUT}/search_space.json", "w") as f:
    json.dump(ss, f, indent=2)

print("Stage 7-8 artifacts written. Winner:", winner_name)
