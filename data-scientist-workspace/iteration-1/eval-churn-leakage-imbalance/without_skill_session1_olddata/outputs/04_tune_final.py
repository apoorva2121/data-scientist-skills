"""Step 4: tune LightGBM with Optuna (time-boxed), decide final model against the noise floor,
evaluate once on the held-out test set, and explain what drives churn."""
import json, time, warnings, importlib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance, partial_dependence

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
m3 = importlib.import_module("03_model_compare")
SEED, CSV, OUT = m3.SEED, m3.CSV, m3.OUT
NUM, CAT = m3.NUM_HONEST, m3.CAT_HONEST
TUNE_TIMEOUT_S = 240

df = m3.load_clean(CSV)
train_df, test_df = train_test_split(df, test_size=0.20, stratify=df["churned"].values, random_state=SEED)
y_tr, y_te = train_df["churned"].values, test_df["churned"].values
print(f"train {len(train_df)} ({y_tr.sum()} pos) / test {len(test_df)} ({y_te.sum()} pos)")
t0 = time.time()

# ----------------------------------------------------------------------------- 1. Optuna tuning (train only)
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
folds = list(cv3.split(train_df, y_tr))


def make_lgbm(p):
    return Pipeline([("pre", m3.make_pre(NUM, CAT, "gbdt")),
                     ("m", lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=4, random_state=SEED, **p))])


def objective(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 31),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1, log=True),
        scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, 3.0, 10.0]),
    )
    aps = []
    for tr, va in folds:
        model = make_lgbm(p).fit(train_df.iloc[tr], y_tr[tr])
        aps.append(average_precision_score(y_tr[va], model.predict_proba(train_df.iloc[va])[:, 1]))
    return float(np.mean(aps))


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=40, timeout=TUNE_TIMEOUT_S)
best = study.best_params
print(f"\nOptuna: {len(study.trials)} trials in {time.time()-t0:.0f}s; best 3-fold PR-AUC {study.best_value:.4f}")
print("best params:", best)

# ----------------------------------------------------------------------------- 2. Decide final model vs noise floor (2x5 CV on train)
cv = m3.RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)


def make_lr(C):
    return Pipeline([("pre", m3.make_pre(NUM, CAT, "linear")),
                     ("m", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


# small regularisation grid for logistic regression (cheap)
lr_grid = {}
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    r = m3.cv_eval(make_lr(C), train_df, y_tr, cv)
    lr_grid[C] = (r.pr_auc.mean(), r.pr_auc.std())
    print(f"logreg C={C:<5} PR-AUC {r.pr_auc.mean():.4f}+-{r.pr_auc.std():.4f}  AUC {r.roc_auc.mean():.4f}")
best_C = max(lr_grid, key=lambda c: lr_grid[c][0])
print("best C:", best_C)

# candidates listed from simplest to most complex
finalists = {
    "logreg": make_lr(best_C),
    "lightgbm_default": m3.candidates(NUM, CAT, 1.0)["lightgbm"],
    "lightgbm_tuned": make_lgbm(best),
}
cv_res = {}
for name, model in finalists.items():
    r = m3.cv_eval(model, train_df, y_tr, cv)
    cv_res[name] = {"mean": r.mean().to_dict(), "std": r.std().to_dict()}
    print(f"{name:18s} AUC {r.roc_auc.mean():.3f}+-{r.roc_auc.std():.3f}  PR-AUC {r.pr_auc.mean():.3f}+-{r.pr_auc.std():.3f}  "
          f"prec@10 {r['precision@10'].mean():.3f}+-{r['precision@10'].std():.3f}  rec@10 {r['recall@10'].mean():.3f}  [{time.time()-t0:.0f}s]")

# Rule: take the simplest model whose CV PR-AUC is within one fold-std (noise floor) of the best CV PR-AUC.
best_pr = max(v["mean"]["pr_auc"] for v in cv_res.values())
noise = np.mean([v["std"]["pr_auc"] for v in cv_res.values()])
final_name = next(n for n in finalists if cv_res[n]["mean"]["pr_auc"] >= best_pr - noise)
print(f"\nbest CV PR-AUC {best_pr:.4f}, noise floor (mean fold std) {noise:.4f} -> final = {final_name}")

# ----------------------------------------------------------------------------- 3. One-shot evaluation on the held-out test set
final = finalists[final_name].fit(train_df, y_tr)
s_te = final.predict_proba(test_df)[:, 1]
test_metrics = m3.evaluate(y_te, s_te)
print("\nTEST:", {k: round(v, 4) for k, v in test_metrics.items()})

rng = np.random.default_rng(SEED)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(y_te), len(y_te))
    if y_te[i].sum() == 0:
        continue
    boot.append(m3.evaluate(y_te[i], s_te[i]))
boot = pd.DataFrame(boot)
ci = {k: (float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))) for k in boot.columns}
print("TEST 95% bootstrap CI:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in ci.items()})

# Baselines on the same test set for context
prior_rate = y_te.mean()
rule = (test_df["contract_type"].eq("Month-to-month").astype(int) * 2 + test_df["support_calls_90d"] / 10
        - test_df["tenure_months"] / 100).values  # simple business rule
rule_metrics = m3.evaluate(y_te, rule)
print(f"baseline random calling: precision@10 = {prior_rate:.3f}; simple rule (M2M + calls - tenure): "
      f"prec@10 {rule_metrics['precision@10']:.3f} rec@10 {rule_metrics['recall@10']:.3f} AUC {rule_metrics['roc_auc']:.3f}")

# Gains table by decile
tbl = pd.DataFrame({"y": y_te, "s": s_te}).sort_values("s", ascending=False).reset_index(drop=True)
tbl["decile"] = (np.arange(len(tbl)) // (len(tbl) / 10)).astype(int) + 1
gains = tbl.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"), avg_score=("s", "mean"), churn_rate=("y", "mean"))
gains["cum_recall"] = gains["churners"].cumsum() / y_te.sum()
gains["lift"] = gains["churn_rate"] / prior_rate
print("\nGains table (test, by predicted-risk decile):\n", gains.round(3).to_string())

# Calibration by decile of predicted probability
cal = tbl.groupby("decile").agg(pred=("s", "mean"), obs=("y", "mean"))
print("\nCalibration (pred vs observed by decile):\n", cal.round(3).T.to_string())

# ----------------------------------------------------------------------------- 4. Drivers
feat = NUM + CAT
pi = permutation_importance(final, test_df[feat], y_te, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=1)
pi_df = pd.DataFrame({"feature": feat, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
print("\nPermutation importance on test (drop in PR-AUC when shuffled):\n", pi_df.round(4).to_string(index=False))

# Group-level permutation: drop whole concept groups to avoid split credit among correlated features
groups = {"tenure/total_charges": ["tenure_months", "total_charges", "avg_monthly_from_total", "signup_year"],
          "price": ["monthly_charges", "charge_vs_history", "charge_per_mb"],
          "support_calls": ["support_calls_90d"],
          "usage": ["avg_daily_usage_mb", "usage_log"],
          "age": ["age"], "contract_type": ["contract_type"], "payment_method": ["payment_method"], "region": ["region"]}
base_ap = average_precision_score(y_te, s_te)
grp = []
for g, cols in groups.items():
    drops = []
    for rep in range(10):
        Xp = test_df.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].iloc[perm].values
        drops.append(base_ap - average_precision_score(y_te, final.predict_proba(Xp)[:, 1]))
    grp.append({"group": g, "pr_auc_drop": np.mean(drops), "std": np.std(drops)})
grp_df = pd.DataFrame(grp).sort_values("pr_auc_drop", ascending=False)
print("\nGroup permutation importance on test:\n", grp_df.round(4).to_string(index=False))

# Direction: partial dependence of the final model on key numeric features (train data as background).
# Computed by hand: set the feature to each grid value for every row, average the predicted probability.
# Derived features that depend on the varied column are recomputed so the PD reflects a real change.
def recompute_derived(X):
    X = X.copy()
    X["avg_monthly_from_total"] = X["total_charges"] / X["tenure_months"]
    X["charge_vs_history"] = X["monthly_charges"] / X["avg_monthly_from_total"]
    X["usage_log"] = np.log1p(X["avg_daily_usage_mb"])
    X["charge_per_mb"] = X["monthly_charges"] / X["avg_daily_usage_mb"]
    return X


pd_out = {}
for f in ["tenure_months", "support_calls_90d", "monthly_charges", "age", "avg_daily_usage_mb"]:
    grid = np.quantile(train_df[f].dropna(), [0.05, 0.25, 0.5, 0.75, 0.95])
    vals = []
    for v in grid:
        Xp = train_df[feat].copy(); Xp[f] = v
        Xp = recompute_derived(Xp)
        vals.append((round(float(v), 2), round(float(final.predict_proba(Xp)[:, 1].mean()), 4)))
    pd_out[f] = dict(vals)
    print(f"PD {f}: {pd_out[f]}")
for f in CAT:
    vals = []
    for v in sorted(train_df[f].unique()):
        Xp = train_df[feat].copy(); Xp[f] = v
        vals.append((v, round(float(final.predict_proba(Xp)[:, 1].mean()), 4)))
    pd_out[f] = dict(vals)
    print(f"PD {f}: {pd_out[f]}")

# Logistic regression coefficients (standardized) as an interpretable cross-check on direction
lr = finalists["logreg"].fit(train_df, y_tr)
names = lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr.named_steps["m"].coef_[0], index=names).sort_values()
print("\nLogistic regression standardized coefficients (most negative / most positive):")
print(pd.concat([coef.head(8), coef.tail(8)]).round(3).to_string())

# ----------------------------------------------------------------------------- 5. Save artifacts
joblib.dump(final, f"{OUT}/final_model.joblib")
pd.DataFrame({"customer_id": test_df["customer_id"].values, "churned": y_te, "score": s_te}).to_csv(f"{OUT}/test_predictions.csv", index=False)
pi_df.to_csv(f"{OUT}/permutation_importance.csv", index=False)
grp_df.to_csv(f"{OUT}/group_importance.csv", index=False)
gains.to_csv(f"{OUT}/gains_table_test.csv")
with open(f"{OUT}/results_final.json", "w") as f:
    json.dump({"final_model": final_name, "best_params": best, "optuna_trials": len(study.trials), "optuna_best_cv3_pr_auc": study.best_value,
               "cv_train_2x5": cv_res, "test_metrics": test_metrics, "test_ci95": ci, "test_prior": float(prior_rate),
               "rule_baseline_test": rule_metrics, "gains": gains.reset_index().to_dict(orient="records"),
               "calibration": cal.reset_index().to_dict(orient="records"), "partial_dependence": {k: {str(a): b for a, b in v.items()} for k, v in pd_out.items()},
               "logreg_coef": coef.round(4).to_dict()}, f, indent=2, default=float)
print(f"\nsaved. total {time.time()-t0:.0f}s")
