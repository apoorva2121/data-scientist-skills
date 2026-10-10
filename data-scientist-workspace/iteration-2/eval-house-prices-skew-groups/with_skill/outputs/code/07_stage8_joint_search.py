"""Stage 8: joint pipeline search (ds-method-comparison Mode 2).

DEVIATION FROM THE TEMPLATE (logged): after Stages 4-6 almost every preprocessing slot collapsed
to a single carried-forward option (missing-value strategy: median for both families; encoding:
one-hot; features: plus_engineered for both families) - only the outlier-treatment slot for the
linear family (none vs winsorize) and the model family itself (Ridge vs LightGBM) still vary. A
custom, narrow Optuna study over this collapsed conditional space was written instead of adapting
the generic scripts/joint_search.py template, to fit the remaining time budget. Budget cut also
logged: 1 seed per trial instead of 3 (deterministic Ridge; LightGBM seeded at a fixed random_state
per trial rather than averaged over seeds).
"""
import json
import sys
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, FrameImputer, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-outliers\scripts")
from outlier_transformers import Winsorizer

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, s = load_train()
train = train.copy()
train["sold_year"] = train["sold_date"].dt.year
train["sold_month"] = train["sold_date"].dt.month
ENG_NUM = NUM_COLS + ["house_age", "total_rooms", "sold_year", "sold_month"]
folds = s.folds
NOISE = 289.57


class AddDomainFeatures(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["house_age"] = X["sold_year"] - X["year_built"]
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        return X


X = train[["sold_year", "sold_month"] + NUM_COLS + CAT_COLS]
y = train["price"].to_numpy()


def build_ridge(alpha, outlier):
    num_steps = []
    if outlier == "winsorize":
        num_steps.append(("wins", ColumnTransformer(
            [("w", Winsorizer(0.01, 0.99), [ENG_NUM.index(c) for c in ["lot_size_sqft", "distance_to_center_km"]])],
            remainder="passthrough")))
    num_steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), ENG_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("feat", AddDomainFeatures()), ("pre", pre),
                      ("model", TransformedTargetRegressor(Ridge(alpha=alpha), func=np.log, inverse_func=np.exp))])


def build_lgbm(n_estimators, num_leaves, lr, min_child, seed):
    imp = FrameImputer(ENG_NUM, CAT_COLS, add_indicator=False)
    model = TransformedTargetRegressor(
        LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=lr,
                      min_child_samples=min_child, random_state=seed, verbosity=-1),
        func=np.log, inverse_func=np.exp)
    return Pipeline([("feat", AddDomainFeatures()), ("prep", imp), ("model", model)])


def cv_mae(pipe):
    maes = []
    for tr, va in folds:
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[va])
        maes.append(mean_absolute_error(y[va], pred))
    return np.array(maes)


trials_log = []

def objective(trial):
    family = trial.suggest_categorical("family", ["ridge", "lightgbm"])
    if family == "ridge":
        alpha = trial.suggest_float("alpha", 0.01, 100, log=True)
        outlier = trial.suggest_categorical("outlier", ["none", "winsorize"])
        pipe = build_ridge(alpha, outlier)
        params = {"alpha": alpha, "outlier": outlier}
    else:
        n_estimators = trial.suggest_int("n_estimators", 100, 400, step=50)
        num_leaves = trial.suggest_int("num_leaves", 7, 63)
        lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
        min_child = trial.suggest_int("min_child", 5, 50)
        pipe = build_lgbm(n_estimators, num_leaves, lr, min_child, seed=0)
        params = {"n_estimators": n_estimators, "num_leaves": num_leaves, "lr": lr, "min_child": min_child}
    maes = cv_mae(pipe)
    trials_log.append({"trial": trial.number, "family": family, **params,
                        "mean_mae": maes.mean(), "sd_mae": maes.std(ddof=1),
                        "fold_maes": maes.round(1).tolist()})
    return maes.mean()


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
N_TRIALS = 25
study.optimize(objective, n_trials=N_TRIALS)
print(f"\n{N_TRIALS} trials in {time.time()-t0:.1f}s")

trials_df = pd.DataFrame(trials_log).sort_values("mean_mae")
trials_df.to_csv(f"{OUT}/search/trials.csv", index=False)
print(trials_df.head(10)[["trial", "family", "mean_mae", "sd_mae"]])

# distinct finalists: best trial per family
finalists = trials_df.loc[trials_df.groupby("family")["mean_mae"].idxmin()].sort_values("mean_mae")
print("\nfinalists (best per family):")
print(finalists[["trial", "family", "mean_mae", "sd_mae"]])
finalists.to_csv(f"{OUT}/search/finalists.csv", index=False)

# 1-SE rule using paired per-fold differences vs the best finalist
best_row = finalists.iloc[0]
best_fold_maes = np.array(best_row["fold_maes"])
pick_row = best_row
ratio = np.mean([len(va) for _, va in folds]) / np.mean([len(tr) for tr, _ in folds])
for _, row in finalists.iterrows():
    if row["trial"] == best_row["trial"]:
        continue
    d = np.array(row["fold_maes"]) - best_fold_maes  # positive = worse (higher MAE)
    n = len(d)
    se = np.sqrt((1 / n + ratio) * d.var(ddof=1))
    print(f"{row['family']}: diff_vs_best={d.mean():.1f}  corrected_se={se:.1f}  "
          f"{'TIE (within 1 SE)' if abs(d.mean()) <= se else 'real difference'}")
    if abs(d.mean()) <= se and row["family"] != pick_row["family"]:
        # tie -> prefer the simpler/cheaper family (ridge: fewer moving parts, faster, more
        # interpretable, per the master skill's tie-break rule)
        if row["family"] == "ridge":
            pick_row = row

print(f"\nPICK: family={pick_row['family']} trial={pick_row['trial']} mean_mae={pick_row['mean_mae']:.1f}")

# --- score pick and best-by-mean on the locked HOLDOUT (Stage 8 only)
df_full = pd.read_csv(f"{OUT}/house_prices_stage1_clean.csv", parse_dates=["sold_date"])
df_full["listing_id"] = df_full["listing_id"].astype(str)
df_full["sold_year"] = df_full["sold_date"].dt.year
df_full["sold_month"] = df_full["sold_date"].dt.month
holdout_df = df_full[s.holdout_mask].reset_index(drop=True)
X_ho = holdout_df[["sold_year", "sold_month"] + NUM_COLS + CAT_COLS]
y_ho = holdout_df["price"].to_numpy()


def build_from_row(row):
    if row["family"] == "ridge":
        return build_ridge(row["alpha"], row["outlier"])
    return build_lgbm(int(row["n_estimators"]), int(row["num_leaves"]), row["lr"], int(row["min_child"]), seed=0)


results = {}
for label, row in [("pick", pick_row), ("best_by_mean", best_row)]:
    pipe = build_from_row(row)
    pipe.fit(X, y)  # fit on train part only, for this holdout check (Stage 9 refits on train+holdout)
    pred = pipe.predict(X_ho)
    mae_ho = mean_absolute_error(y_ho, pred)
    se_ho = np.abs(y_ho - pred).std(ddof=1) / np.sqrt(len(y_ho))
    results[label] = {"family": row["family"], "params": {k: row[k] for k in row.index if k not in
                       ("trial", "mean_mae", "sd_mae", "fold_maes")},
                       "cv_mean_mae": float(row["mean_mae"]), "cv_sd_mae": float(row["sd_mae"]),
                       "holdout_mae": float(mae_ho), "holdout_se": float(se_ho)}
    print(f"{label}: family={row['family']} CV MAE={row['mean_mae']:.1f} -> HOLDOUT MAE={mae_ho:.1f} +/- {se_ho:.1f}")

gap = abs(results["pick"]["holdout_mae"] - results["pick"]["cv_mean_mae"])
gap_verdict = "OK (within ~2 SE of CV mean)" if gap <= 2 * results["pick"]["holdout_se"] + results["pick"]["cv_sd_mae"] else "WATCH: holdout gap exceeds ~2 SE, possible overfit/drift"
print("holdout-gap check:", gap_verdict)

json.dump({"n_trials": N_TRIALS, "elapsed_s": round(time.time() - t0, 1),
           "results": results, "holdout_gap_check": gap_verdict,
           "pick_trial": int(pick_row["trial"]), "best_by_mean_trial": int(best_row["trial"])},
          open(f"{OUT}/search/result.json", "w"), indent=2, default=str)
print("wrote", f"{OUT}/search/result.json")
