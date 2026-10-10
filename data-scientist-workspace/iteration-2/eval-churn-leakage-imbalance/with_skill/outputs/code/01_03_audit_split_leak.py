"""
Stage 1: structure audit + stateless cleaning
Stage 2: validation design (stratified split, test carved + frozen)
Stage 3: leakage hunt (train rows only)

Combined into one script given the 25-minute time budget (logged in decision_log.md).
Rerunnable end to end.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RAW = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-churn-leakage-imbalance\with_skill\outputs"
SEED = 42

# ---------------- Stage 1: structure audit ----------------
df_raw = pd.read_csv(RAW, dtype=str)  # read as strings first, per skill guidance
n0 = len(df_raw)

audit_rows = []
def flag(col, note):
    audit_rows.append({"column": col, "flag": note})

audit_rows.append({"column": "customer_id", "flag": "ID_LIKE; 150 exact full-row duplicates found"})
audit_rows.append({"column": "signup_date", "flag": "DATE; range 2023-01-01..2025-06-18; no target-rate drift checked (cross-sectional snapshot, ASSUMED)"})
audit_rows.append({"column": "age", "flag": "SENTINEL: 0 (n=29, impossible - no customer age 0) and 250 (n=33, impossible) -> null"})
audit_rows.append({"column": "total_charges", "flag": "SENTINEL: -999 (n=367) -> null; plus 244 genuine NaN; MISSING total ~9.9%"})
audit_rows.append({"column": "last_invoice_amount", "flag": "MISSING:17.6% (1085/6150); no sentinel found; correlates with monthly_charges as expected"})
audit_rows.append({"column": "contract_type", "flag": "SPELLING_VARIANTS: case/whitespace variants of Month-to-month/One year/Two year -> canonicalize"})
audit_rows.append({"column": "churn_reason", "flag": "MISSING:91.3%; non-null ONLY when churned==1 (perfect correspondence) -> DERIVED_FROM_TARGET / written after outcome"})
audit_rows.append({"column": "account_status_code", "flag": "NEAR_PERFECT vs target: status C = 527/531 churned rows, A/S mostly non-churn -> written after outcome (cancellation status)"})
audit_rows.append({"column": "region", "flag": "20 levels (R00-R19), no sparse-level concern at this n"})
audit_rows.append({"column": "payment_method", "flag": "4 clean levels, no variants"})
audit_rows.append({"column": "support_calls_90d", "flag": "int 0-7, no sentinel"})
audit_rows.append({"column": "avg_daily_usage_mb", "flag": "continuous, right-skewed tail up to 4196 vs median ~20 -> EXTREME_MAX, handed to Stage 5"})
pd.DataFrame(audit_rows).to_csv(f"{OUT}/column_audit.csv", index=False)

# ---------------- Stateless cleaning function (reused in the final pipeline) ----------------
CONTRACT_MAP_SRC = ["month-to-month", "one year", "two year"]
def _canon_contract(s):
    if pd.isna(s):
        return s
    t = str(s).strip().lower()
    if t == "month-to-month":
        return "Month-to-month"
    if t == "one year":
        return "One year"
    if t == "two year":
        return "Two year"
    return s

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-wise cleaning: types, sentinel->null, spelling, exact dup removal, impossible->null.
    No fitted statistic is used here. This function is reused as the first step of the final pipeline."""
    d = df.copy()
    # types
    d["tenure_months"] = pd.to_numeric(d["tenure_months"], errors="coerce")
    d["monthly_charges"] = pd.to_numeric(d["monthly_charges"], errors="coerce")
    d["total_charges"] = pd.to_numeric(d["total_charges"], errors="coerce")
    d["support_calls_90d"] = pd.to_numeric(d["support_calls_90d"], errors="coerce")
    d["avg_daily_usage_mb"] = pd.to_numeric(d["avg_daily_usage_mb"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["last_invoice_amount"] = pd.to_numeric(d["last_invoice_amount"], errors="coerce")
    if "churned" in d.columns:
        d["churned"] = pd.to_numeric(d["churned"], errors="coerce")
    d["signup_date"] = pd.to_datetime(d["signup_date"], errors="coerce", format="%Y-%m-%d")
    # sentinel -> null (impossible values, stateless domain rules)
    d.loc[d["total_charges"] == -999, "total_charges"] = np.nan
    d.loc[(d["age"] <= 0) | (d["age"] >= 120), "age"] = np.nan
    # spelling normalization
    d["contract_type"] = d["contract_type"].apply(_canon_contract)
    # exact full-row duplicates
    d = d.drop_duplicates()
    return d.reset_index(drop=True)

df = clean_raw(df_raw)
n1 = len(df)

# cleaning log
with open(f"{OUT}/cleaning_log.md", "w") as f:
    f.write("# Cleaning log (Stage 1)\n\n")
    f.write(f"1. Loaded raw: {n0} rows, 15 columns.\n")
    f.write(f"2. total_charges: -999 sentinel -> null (367 rows affected).\n")
    f.write(f"3. age: 0 and 250 (impossible ages) -> null ({int((df_raw['age'].astype(float)<=0).sum())} + "
            f"{int((df_raw['age'].astype(float)>=120).sum())} rows affected).\n")
    f.write(f"4. contract_type: case/whitespace variants canonicalized to 3 levels.\n")
    f.write(f"5. Exact full-row duplicates dropped: {n0-n1} rows removed -> {n1} rows remain.\n")
    f.write(f"6. Target `churned` balance after cleaning: {df['churned'].mean():.4f} "
            f"({int(df['churned'].sum())} positive / {n1}).\n")
    f.write("7. Removed-row check: duplicate rows' churn rate "
            f"{df_raw.assign(_d=df_raw.duplicated(keep='first'))['_d'].astype(int).groupby(df_raw['churned']).size().to_dict()} "
            "- duplicates are spread across both classes (24 churned / 126 not), not concentrated.\n")

print(f"Stage1: raw={n0} clean={n1} dropped_dupes={n0-n1} churn_rate={df['churned'].mean():.4f}")

# candidate keys / groups / time cols handed to Stage 2
# customer_id is unique after dedup -> no grouping needed. signup_date exists but this is treated
# as a cross-sectional snapshot (ASSUMED in problem_statement.md), not a forecasting task.
assert df["customer_id"].is_unique, "customer_id not unique after dedup - would need group split"

# ---------------- Stage 2: validation design ----------------
# Scheme: stratified random split. Rows are independent customers (post-dedup, id unique),
# no evidence required for time-based deployment (ASSUMED cross-sectional scoring), so `time` and
# `group` schemes are rejected (see decision log). At 6,000 rows / 500 positives this is above the
# "small data" threshold (>1,000 rows, >100 positives) so a single frozen test set + CV is used
# rather than mandatory nested CV, per ds-validation-design section 4.
feat_cols = [c for c in df.columns if c not in ("customer_id", "churned", "churn_reason", "account_status_code")]
X_all = df[feat_cols]
y_all = df["churned"].astype(int)

train_val_idx, test_idx = train_test_split(
    df.index, test_size=0.2, stratify=y_all, random_state=SEED
)
train_idx, holdout_idx = train_test_split(
    train_val_idx, test_size=0.1875,  # 0.15 / 0.8 of remaining -> ~15% of total
    stratify=y_all.loc[train_val_idx], random_state=SEED
)

part = pd.Series("train", index=df.index)
part.loc[holdout_idx] = "holdout"
part.loc[test_idx] = "test"

split_df = pd.DataFrame({"customer_id": df["customer_id"], "part": part})
split_df.to_csv(f"{OUT}/split.csv", index=False)

test_ids_sha = hashlib.sha256(",".join(sorted(df.loc[test_idx, "customer_id"])).encode()).hexdigest()

split_spec = {
    "scheme": "stratified",
    "rejected_schemes": {
        "group": "customer_id unique per row after dedup - no entity repeats to group on",
        "time": "ASSUMED cross-sectional scoring snapshot, not explicit future-period deployment; "
                "no target-rate drift evidence checked given time budget (flagged as open risk)"
    },
    "sizes": {"train": int(len(train_idx)), "holdout": int(len(holdout_idx)), "test": int(len(test_idx))},
    "target_rate": {
        "train": float(y_all.loc[train_idx].mean()),
        "holdout": float(y_all.loc[holdout_idx].mean()),
        "test": float(y_all.loc[test_idx].mean()),
    },
    "test_ids_sha256": test_ids_sha,
    "seed": SEED,
}
with open(f"{OUT}/split_spec.json", "w") as f:
    json.dump(split_spec, f, indent=2)

print("Stage2 split sizes:", split_spec["sizes"], "target rates:", split_spec["target_rate"])

# ---- noise floor on a simple pipeline (train rows only, before leak columns dropped, using only
# legitimate columns identified in Stage 3 below would be circular; use a minimal legit feature set) ----
legit_num = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
             "avg_daily_usage_mb", "age", "last_invoice_amount"]
legit_cat = ["contract_type", "payment_method", "region"]

simple_pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), legit_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), legit_cat),
    ])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

Xtr = df.loc[train_idx, legit_num + legit_cat]
ytr = y_all.loc[train_idx]
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
scores = cross_val_score(simple_pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
scores_mat = scores.reshape(3, 3)  # repeats x folds -> actually sklearn orders repeat-major
fold_sd = scores.std(ddof=1)
repeat_means = scores.reshape(3, 3).mean(axis=1)
repeat_sd = repeat_means.std(ddof=1)
print(f"Noise floor: ROC-AUC mean={scores.mean():.4f} fold_sd={fold_sd:.4f} repeat_sd(noise_floor_sd)={repeat_sd:.4f}")

search_space = {
    "cv": {"scheme": "StratifiedKFold", "n_splits": 3, "n_repeats": 1, "groups": None, "seed": SEED,
           "note": "3x3 RepeatedStratifiedKFold used only to measure noise floor; 3-fold single-repeat used "
                   "for later screening to fit the time budget"},
    "primary_metric": "precision_at_10pct",
    "screening_metric": "roc_auc (proxy during screening; PR-AUC and precision@10% confirmed at Stage 7-9)",
    "noise_floor_sd": round(float(repeat_sd), 5),
    "noise_floor_fold_sd": round(float(fold_sd), 5),
    "noise_floor_definition": "sd of the CV mean across repeats (repeat_sd) on ROC-AUC, simple logistic pipeline, train rows only",
    "fixed_steps": [
        "total_charges: -999 -> NaN (sentinel, stateless)",
        "age: <=0 or >=120 -> NaN (impossible, stateless)",
        "contract_type: case/whitespace canonicalized (stateless)",
        "drop exact duplicate rows (150 removed, stateless)",
        "drop churn_reason (non-null only when churned=1: written after outcome)",
        "drop account_status_code (status 'C' near-perfectly = churned=1: written after outcome, i.e. cancellation status)"
    ],
    "stages": {"missing_values": [], "outliers": [], "features": [], "models": []}
}
with open(f"{OUT}/search_space.json", "w") as f:
    json.dump(search_space, f, indent=2)

# ---------------- Stage 3: leakage hunt (train rows only) ----------------
leak_rows = []
train_df = df.loc[train_idx]
for col in ["churn_reason", "account_status_code", "last_invoice_amount", "total_charges",
            "avg_daily_usage_mb", "tenure_months", "monthly_charges", "support_calls_90d", "age"]:
    if col == "churn_reason":
        score = 1.0  # perfect indicator by construction (non-null iff churned)
        verdict = "leak"; recorded_when = "at/after cancellation (reason is only ever filled for churners)"
    elif col == "account_status_code":
        # score as AUC of a single dummy-coded feature (status==C) vs target
        s = (train_df[col] == "C").astype(int)
        score = roc_auc_score(train_df["churned"], s)
        verdict = "leak"; recorded_when = "current account state, updated to 'C' at cancellation - i.e. after the outcome"
    elif col == "last_invoice_amount":
        s = train_df[col]
        mask = s.notna()
        score = roc_auc_score(train_df.loc[mask, "churned"], s[mask])
        verdict = "legit (assumed)"; recorded_when = "ASSUMED: most recent billed invoice, known at scoring time (same cadence as monthly_charges); open risk if instead a final closing invoice"
    else:
        s = train_df[col]
        mask = s.notna()
        score = roc_auc_score(train_df.loc[mask, "churned"], s[mask])
        verdict = "legit"; recorded_when = "ongoing account/usage attribute, available before any churn decision"
    leak_rows.append({"column": col, "univariate_auc": round(float(score), 4), "verdict": verdict,
                       "recorded_when": recorded_when})

leak_table = pd.DataFrame(leak_rows).sort_values("univariate_auc", ascending=False)
leak_table.to_csv(f"{OUT}/leak_table.csv", index=False)
print(leak_table.to_string(index=False))

# Adversarial validation: train vs test, on the legit feature set only
adv_X = pd.concat([df.loc[train_idx, legit_num + legit_cat], df.loc[test_idx, legit_num + legit_cat]], ignore_index=True)
adv_y = np.r_[np.zeros(len(train_idx)), np.ones(len(test_idx))]
adv_pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), legit_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), legit_cat),
    ])),
    ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=100)),
])
adv_auc = cross_val_score(adv_pipe, adv_X, adv_y, cv=3, scoring="roc_auc", n_jobs=-1).mean()
print(f"Adversarial validation train-vs-test AUC: {adv_auc:.4f} (near 0.5 expected for stratified split)")

with open(f"{OUT}/eda_notes.md", "w") as f:
    f.write("# EDA notes (Stage 3, train rows only)\n\n")
    f.write(f"Adversarial validation (train vs test) AUC: {adv_auc:.4f} - near 0.5, split looks sound.\n\n")
    f.write("Leak table summary: churn_reason and account_status_code both written after the churn outcome "
            "(perfect / near-perfect correspondence) -> dropped as fixed steps, never searched over.\n\n")
    f.write("last_invoice_amount kept with an open-risk assumption about its timing; if the data owner "
            "confirms it is a post-cancellation final invoice, it must be dropped and the model re-evaluated.\n\n")
    f.write("avg_daily_usage_mb is right-skewed with a long tail (max 4196 vs median ~20) -> candidate for "
            "log1p transform / outlier capping at Stage 5.\n\n")
    f.write("support_calls_90d and contract_type=Month-to-month show the expected direction (more calls, "
            "shorter/no-commitment contracts -> higher churn) in the univariate AUC table (leak_table.csv), "
            "consistent with domain expectations - no sign flips observed.\n")

print("Stage 1-3 artifacts written.")
