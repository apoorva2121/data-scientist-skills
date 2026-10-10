"""
Stage 4: missing values (diagnose, screen candidates, carry forward)
Stage 5: outliers (profile, verdict, screen candidates)
Stage 6: feature engineering (ledger, candidates, ablation, carry forward)

Combined given the 25-minute time budget. Screens on Stage 2 train folds with two proxies
(regularised linear, small GBDT), ROC-AUC (established as the screening metric at Stage 2).
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

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

df_raw = pd.read_csv(RAW, dtype=str)
df = clean_raw(df_raw)
df = df.drop(columns=["churn_reason", "account_status_code"])  # Stage 3 fixed drop

split = pd.read_csv(f"{OUT}/split.csv")
df = df.merge(split, on="customer_id")
train = df[df["part"] == "train"].reset_index(drop=True)
y = train["churned"].astype(int)

NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
       "avg_daily_usage_mb", "age", "last_invoice_amount"]
CAT = ["contract_type", "payment_method", "region"]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

# =========================== Stage 4: Missing values ===========================
miss_profile = []
for c in NUM:
    share = train[c].isna().mean()
    if share == 0:
        continue
    ind = train[c].isna().astype(int)
    # is missingness predictable from other numeric cols? (quick CV AUC of a small model on indicator)
    other = [x for x in NUM if x != c]
    Xo = train[other].fillna(train[other].median())
    try:
        pred_auc = cross_val_score(HistGradientBoostingClassifier(random_state=SEED, max_iter=50),
                                    Xo, ind, cv=3, scoring="roc_auc").mean()
    except Exception:
        pred_auc = np.nan
    # does the indicator predict the target?
    from sklearn.metrics import roc_auc_score
    target_auc = roc_auc_score(y, ind)
    miss_profile.append({"column": c, "missing_share": round(share, 4),
                          "indicator_predicts_other_cols_auc": round(float(pred_auc), 4),
                          "indicator_predicts_target_auc": round(float(target_auc), 4)})
mp = pd.DataFrame(miss_profile)
mp["mechanism_verdict"] = mp.apply(
    lambda r: "informative" if abs(r["indicator_predicts_target_auc"] - 0.5) > 0.03
    else ("MAR-like" if r["indicator_predicts_other_cols_auc"] > 0.6 else "MCAR-like"), axis=1)
mp.to_csv(f"{OUT}/missing_profile.csv", index=False)
print(mp.to_string(index=False))

def make_prep(num_impute="median", add_indicator=False, cat_impute="most_frequent"):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy=num_impute, add_indicator=add_indicator))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy=cat_impute)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])

def linear_pipe(prep):
    return Pipeline([("prep", prep), ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def gbdt_pipe_native():
    # HistGradientBoostingClassifier handles NaN natively; category via ordinal-ish OHE (small card) kept dense
    prep = ColumnTransformer([("num", "passthrough", NUM),
                               ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                  ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    return Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=150))])

def gbdt_pipe_imputed(add_indicator=False):
    prep = make_prep("median", add_indicator=add_indicator)
    return Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=150))])

Xtr = train[NUM + CAT]
results = {}
results["linear__median"] = cross_val_score(linear_pipe(make_prep("median", False)), Xtr, y, cv=cv, scoring="roc_auc")
results["linear__median_indicator"] = cross_val_score(linear_pipe(make_prep("median", True)), Xtr, y, cv=cv, scoring="roc_auc")
results["gbdt__native_nan"] = cross_val_score(gbdt_pipe_native(), Xtr, y, cv=cv, scoring="roc_auc")
results["gbdt__median_indicator"] = cross_val_score(gbdt_pipe_imputed(True), Xtr, y, cv=cv, scoring="roc_auc")

for k, v in results.items():
    print(f"{k}: {v.mean():.4f} +/- {v.std(ddof=1):.4f}")

NOISE = 0.0024  # repeat_sd from Stage 2

# =========================== Stage 5: Outliers ===========================
outlier_rows = []
from scipy import stats as sps
for c in NUM:
    s = train[c].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    frac_out = ((s < lo) | (s > hi)).mean()
    skew = s.skew()
    extreme_lift = np.nan
    if frac_out > 0:
        ext_mask = train[c].notna() & ((train[c] < lo) | (train[c] > hi))
        if ext_mask.sum() >= 5:
            extreme_lift = y[ext_mask].mean() / y.mean()
    outlier_rows.append({"column": c, "frac_beyond_1.5IQR": round(float(frac_out), 4),
                          "skew": round(float(skew), 3), "target_lift_in_extremes": round(float(extreme_lift), 3) if not np.isnan(extreme_lift) else None})
op = pd.DataFrame(outlier_rows)
def verdict(r):
    if r["skew"] > 1 and r["frac_beyond_1.5IQR"] > 0.03:
        return "genuine tail - transform candidate (linear only)"
    if r["target_lift_in_extremes"] is not None and abs(r["target_lift_in_extremes"] - 1) > 0.3:
        return "signal - keep, no removal"
    return "no material outlier concern"
op["verdict"] = op.apply(verdict, axis=1)
op.to_csv(f"{OUT}/outlier_profile.csv", index=False)
print(op.to_string(index=False))

# screen log1p(avg_daily_usage_mb) for linear proxy only (trees invariant to monotonic transforms - skill red flag)
class Log1pCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X

def make_prep_log(cols_to_log):
    pre_log = FunctionTransformer(lambda X: (lambda Xc: (Xc.assign(**{c: np.log1p(Xc[c].clip(lower=0)) for c in cols_to_log})))(X.copy()))
    num_pipe = Pipeline([("log", pre_log if cols_to_log else "passthrough"), ("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])

lin_none = results["linear__median_indicator"]
lin_log = cross_val_score(linear_pipe(make_prep_log(["avg_daily_usage_mb"])), Xtr, y, cv=cv, scoring="roc_auc")
print(f"linear__log1p_usage: {lin_log.mean():.4f} +/- {lin_log.std(ddof=1):.4f}  vs none {lin_none.mean():.4f}")

# =========================== Stage 6: Feature engineering ===========================
def add_engineered(X):
    X = X.copy()
    X["charge_gap"] = X["monthly_charges"] * X["tenure_months"] - X["total_charges"]
    X["is_new_customer"] = (X["tenure_months"] <= 3).astype(int)
    return X

train_fe = add_engineered(train[NUM + CAT])
NUM_FE = NUM + ["charge_gap", "is_new_customer"]

def gbdt_pipe_fe():
    prep = ColumnTransformer([("num", "passthrough", NUM_FE),
                               ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                  ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    return Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=150))])

def linear_pipe_fe():
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_FE),
                               ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                  ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

X_fe = pd.concat([train_fe, train[[]]], axis=1)
gbdt_fe_scores = cross_val_score(gbdt_pipe_fe(), X_fe, y, cv=cv, scoring="roc_auc")
lin_fe_scores = cross_val_score(linear_pipe_fe(), X_fe, y, cv=cv, scoring="roc_auc")
print(f"gbdt__plus_engineered: {gbdt_fe_scores.mean():.4f} +/- {gbdt_fe_scores.std(ddof=1):.4f} (vs gbdt native {results['gbdt__native_nan'].mean():.4f})")
print(f"linear__plus_engineered: {lin_fe_scores.mean():.4f} +/- {lin_fe_scores.std(ddof=1):.4f} (vs linear median_ind {lin_none.mean():.4f})")

feature_ledger = pd.DataFrame([
    {"feature": "charge_gap", "group": "billing_consistency", "inputs": "monthly_charges,tenure_months,total_charges",
     "transformation": "monthly_charges*tenure_months - total_charges", "available_at_prediction": "yes: all 3 inputs are current-state fields",
     "fitted_in_fold": "no (stateless arithmetic)", "families": "linear,gbdt"},
    {"feature": "is_new_customer", "group": "tenure", "inputs": "tenure_months", "transformation": "tenure_months<=3",
     "available_at_prediction": "yes", "fitted_in_fold": "no (stateless)", "families": "linear,gbdt"},
    {"feature": "log1p(avg_daily_usage_mb)", "group": "usage_transform", "inputs": "avg_daily_usage_mb",
     "transformation": "log1p", "available_at_prediction": "yes", "fitted_in_fold": "no (stateless, no learned param)",
     "families": "linear only (trees invariant to monotonic transforms)"},
    {"feature": "missing indicators (total_charges, last_invoice_amount)", "group": "missingness",
     "inputs": "total_charges,last_invoice_amount", "transformation": "SimpleImputer(add_indicator=True)",
     "available_at_prediction": "yes", "fitted_in_fold": "yes (imputer)", "families": "linear,gbdt"},
])
feature_ledger.to_csv(f"{OUT}/feature_ledger.csv", index=False)

# =========================== Update search_space.json ===========================
with open(f"{OUT}/search_space.json") as f:
    ss = json.load(f)

ss["stages"]["missing_values"] = [
    {"name": "median_indicator", "applies_to": ["linear"],
     "proxy_scores": {"linear": [round(float(results['linear__median_indicator'].mean()),4), round(float(results['linear__median_indicator'].std(ddof=1)),4)]}},
    {"name": "native_nan", "applies_to": ["gbdt"],
     "proxy_scores": {"gbdt": [round(float(results['gbdt__native_nan'].mean()),4), round(float(results['gbdt__native_nan'].std(ddof=1)),4)]}},
    {"name": "median_indicator", "applies_to": ["gbdt"],
     "proxy_scores": {"gbdt": [round(float(results['gbdt__median_indicator'].mean()),4), round(float(results['gbdt__median_indicator'].std(ddof=1)),4)]}},
]
ss["stages"]["outliers"] = [
    {"name": "none", "applies_to": ["linear", "gbdt"], "proxy_scores": {"linear": [round(float(lin_none.mean()),4), round(float(lin_none.std(ddof=1)),4)]}},
    {"name": "log1p_avg_daily_usage_mb", "applies_to": ["linear"], "proxy_scores": {"linear": [round(float(lin_log.mean()),4), round(float(lin_log.std(ddof=1)),4)]}},
]
ss["stages"]["features"] = [
    {"name": "base_features_only", "applies_to": ["linear", "gbdt"],
     "proxy_scores": {"linear": [round(float(lin_none.mean()),4), round(float(lin_none.std(ddof=1)),4)],
                       "gbdt": [round(float(results['gbdt__median_indicator'].mean()),4), round(float(results['gbdt__median_indicator'].std(ddof=1)),4)]}},
    {"name": "base_plus_engineered", "applies_to": ["linear", "gbdt"],
     "proxy_scores": {"linear": [round(float(lin_fe_scores.mean()),4), round(float(lin_fe_scores.std(ddof=1)),4)],
                       "gbdt": [round(float(gbdt_fe_scores.mean()),4), round(float(gbdt_fe_scores.std(ddof=1)),4)]}},
]
with open(f"{OUT}/search_space.json", "w") as f:
    json.dump(ss, f, indent=2)

print("Stage 4-6 artifacts written.")
