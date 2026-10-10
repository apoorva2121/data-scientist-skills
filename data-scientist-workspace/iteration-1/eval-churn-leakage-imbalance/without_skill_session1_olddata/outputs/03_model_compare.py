"""Step 3: clean data, build leak-free features, compare model families under repeated stratified CV,
and quantify how much the leakage columns inflate the score."""
import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
SEED = 42
CSV = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\churn.csv"
OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-1\eval-churn-leakage-imbalance\without_skill\outputs"


# ----------------------------------------------------------------------------- cleaning
def load_clean(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates()                       # 150 exact duplicate rows
    assert df["customer_id"].is_unique
    df["contract_type"] = (df["contract_type"].str.strip().str.lower()
                           .map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}))
    assert df["contract_type"].notna().all()
    df.loc[df["total_charges"] == -999, "total_charges"] = np.nan     # sentinel -> missing
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan       # 0 and 250 are invalid
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    # derived, all computable before churn happens
    df["avg_monthly_from_total"] = df["total_charges"] / df["tenure_months"]
    df["charge_vs_history"] = df["monthly_charges"] / df["avg_monthly_from_total"]   # >1 => paying more now than historically
    df["usage_log"] = np.log1p(df["avg_daily_usage_mb"])
    df["charge_per_mb"] = df["monthly_charges"] / df["avg_daily_usage_mb"]
    df["signup_year"] = df["signup_date"].dt.year
    df["last_invoice_missing"] = df["last_invoice_amount"].isna().astype(int)
    print(f"rows: {n0} -> {len(df)} after dedupe; churn rate {df.churned.mean():.4f} ({df.churned.sum()} churners)")
    return df


NUM_HONEST = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb",
              "usage_log", "age", "avg_monthly_from_total", "charge_vs_history", "charge_per_mb", "signup_year"]
CAT_HONEST = ["contract_type", "payment_method", "region"]
# Leakage variants for the audit
LEAK_STATUS = ["account_status_code"]
LEAK_REASON = ["churn_reason"]
LEAK_INVOICE = ["last_invoice_amount", "last_invoice_missing"]


def make_pre(num, cat, kind):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "linear":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    elif kind == "rf":
        num_pipe = SimpleImputer(strategy="median", add_indicator=True)
    else:  # gbdt: keep NaN, models handle it natively
        num_pipe = "passthrough"
    return ColumnTransformer([("num", num_pipe, num), ("cat", ohe, cat)], remainder="drop")


# ----------------------------------------------------------------------------- metrics
def topk(y, s, frac=0.10):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-s)[:k]
    hits = y[idx].sum()
    return {"precision@10": hits / k, "recall@10": hits / y.sum(), "lift@10": (hits / k) / y.mean()}


def evaluate(y, s):
    m = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "brier": brier_score_loss(y, s)}
    m.update(topk(y, s))
    return m


def cv_eval(model, X, y, cv):
    rows = []
    for tr, va in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        s = model.predict_proba(X.iloc[va])[:, 1]
        rows.append(evaluate(y[va], s))
    return pd.DataFrame(rows)


def candidates(num, cat, pos_w):
    return {
        "baseline_prior": Pipeline([("pre", make_pre(num, cat, "linear")), ("m", DummyClassifier(strategy="prior"))]),
        "logreg": Pipeline([("pre", make_pre(num, cat, "linear")),
                            ("m", LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))]),
        "random_forest": Pipeline([("pre", make_pre(num, cat, "rf")),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                                class_weight="balanced_subsample", n_jobs=4, random_state=SEED))]),
        "hist_gbdt": Pipeline([("pre", make_pre(num, cat, "gbdt")),
                               ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                                    min_samples_leaf=30, l2_regularization=1.0, random_state=SEED))]),
        "lightgbm": Pipeline([("pre", make_pre(num, cat, "gbdt")),
                              ("m", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                                       verbose=-1, n_jobs=4, random_state=SEED))]),
        "lightgbm_weighted": Pipeline([("pre", make_pre(num, cat, "gbdt")),
                                       ("m", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                                                scale_pos_weight=pos_w, verbose=-1, n_jobs=4, random_state=SEED))]),
        "xgboost": Pipeline([("pre", make_pre(num, cat, "gbdt")),
                             ("m", xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                     eval_metric="logloss", random_state=SEED, n_jobs=4))]),
    }


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    df = load_clean(CSV)
    y_all = df["churned"].values
    train_df, test_df = train_test_split(df, test_size=0.20, stratify=y_all, random_state=SEED)
    y_tr, y_te = train_df["churned"].values, test_df["churned"].values
    print(f"train {len(train_df)} ({y_tr.sum()} pos) / test {len(test_df)} ({y_te.sum()} pos)")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    pos_w = (1 - y_tr.mean()) / y_tr.mean()

    results = {}
    t0 = time.time()
    # --- honest feature set: full comparison
    print("\n=== HONEST features (leak columns removed) : 2x5 stratified CV on train ===")
    for name, model in candidates(NUM_HONEST, CAT_HONEST, pos_w).items():
        r = cv_eval(model, train_df, y_tr, cv)
        results[f"honest/{name}"] = {"mean": r.mean().to_dict(), "std": r.std().to_dict()}
        print(f"{name:20s} AUC {r.roc_auc.mean():.3f}+-{r.roc_auc.std():.3f}  PR-AUC {r.pr_auc.mean():.3f}+-{r.pr_auc.std():.3f}  "
              f"prec@10 {r['precision@10'].mean():.3f}+-{r['precision@10'].std():.3f}  rec@10 {r['recall@10'].mean():.3f}  "
              f"lift {r['lift@10'].mean():.2f}  brier {r.brier.mean():.4f}   [{time.time()-t0:.0f}s]")

    # --- leakage audit: same LightGBM, adding each leak column back in
    cv_audit = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    print("\n=== LEAKAGE AUDIT (lightgbm, 1x5 stratified CV on train) ===")
    audits = {
        "honest": (NUM_HONEST, CAT_HONEST),
        "+last_invoice(missingness)": (NUM_HONEST + LEAK_INVOICE, CAT_HONEST),
        "+account_status_code": (NUM_HONEST, CAT_HONEST + LEAK_STATUS),
        "+churn_reason": (NUM_HONEST, CAT_HONEST + LEAK_REASON),
        "all_columns_naive": (NUM_HONEST + LEAK_INVOICE, CAT_HONEST + LEAK_STATUS + LEAK_REASON),
    }
    for label, (num, cat) in audits.items():
        model = candidates(num, cat, pos_w)["lightgbm"]
        r = cv_eval(model, train_df, y_tr, cv_audit)
        results[f"audit/{label}"] = {"mean": r.mean().to_dict(), "std": r.std().to_dict()}
        print(f"{label:28s} AUC {r.roc_auc.mean():.3f}  PR-AUC {r.pr_auc.mean():.3f}  "
              f"prec@10 {r['precision@10'].mean():.3f}  rec@10 {r['recall@10'].mean():.3f}")

    with open(f"{OUT}/results_model_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s")
