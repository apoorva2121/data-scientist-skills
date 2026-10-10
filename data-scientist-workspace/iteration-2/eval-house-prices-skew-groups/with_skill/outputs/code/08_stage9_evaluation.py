"""Stage 9: evaluation + sign-off. Test set scored once. Split-conformal implemented by hand
(MAPIE not installed, per environment constraint) using the locked HOLDOUT as the calibration set.

Design note (logged in decision_log.md): the shipped point-prediction model is fit on TRAIN ONLY
(2951 rows), not train+holdout, so the holdout residuals used for the conformal margin come from a
model that has genuinely never seen those rows - the textbook split-conformal recipe. This trades
~18% less training data for a calibration set whose coverage guarantee is honest, which the task
explicitly asked for ("some idea of how far off it might be for each estimate").
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
from cv_folds import load_split, verify_test_untouched
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-outliers\scripts")
from outlier_transformers import Winsorizer
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-evaluation\scripts")
from final_eval import bootstrap_ci, slice_table, shuffled_label_check, interval_coverage

assert verify_test_untouched(f"{OUT}/split") is True, "test set hash changed since Stage 2 - STOP"

train, s = load_train()
train = train.copy()
train["sold_year"] = train["sold_date"].dt.year
train["sold_month"] = train["sold_date"].dt.month
ENG_NUM = NUM_COLS + ["house_age", "total_rooms", "sold_year", "sold_month"]

df_full = pd.read_csv(f"{OUT}/house_prices_stage1_clean.csv", parse_dates=["sold_date"])
df_full["listing_id"] = df_full["listing_id"].astype(str)
df_full["sold_year"] = df_full["sold_date"].dt.year
df_full["sold_month"] = df_full["sold_date"].dt.month
holdout_df = df_full[s.holdout_mask].reset_index(drop=True)
test_df = df_full[s.test_mask].reset_index(drop=True)

FEATS = ["sold_year", "sold_month"] + NUM_COLS + CAT_COLS
X_tr, y_tr = train[FEATS], train["price"].to_numpy()
X_ho, y_ho = holdout_df[FEATS], holdout_df["price"].to_numpy()
X_te, y_te = test_df[FEATS], test_df["price"].to_numpy()


class AddDomainFeatures(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["house_age"] = X["sold_year"] - X["year_built"]
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        return X


def build_final(alpha=0.011545):
    pre = ColumnTransformer([
        ("num", Pipeline([("wins", ColumnTransformer(
                              [("w", Winsorizer(0.01, 0.99), [ENG_NUM.index(c) for c in ["lot_size_sqft", "distance_to_center_km"]])],
                              remainder="passthrough")),
                           ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ENG_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("feat", AddDomainFeatures()), ("pre", pre),
                      ("model", TransformedTargetRegressor(Ridge(alpha=alpha), func=np.log, inverse_func=np.exp))])


final_pipe = build_final()
final_pipe.fit(X_tr, y_tr)

# --- Step 4: split-conformal, calibrated on the holdout, in LOG space (handles heteroscedasticity)
ALPHA = 0.10  # 90% target coverage
pred_ho = final_pipe.predict(X_ho)
nonconformity = np.abs(np.log(y_ho) - np.log(pred_ho))
n = len(nonconformity)
q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
q = float(np.quantile(nonconformity, q_level, method="higher"))
print(f"conformal margin (log space), q={q:.4f} at level {q_level:.4f}, n_calib={n}")

pred_te = final_pipe.predict(X_te)
lower_te = np.exp(np.log(pred_te) - q)
upper_te = np.exp(np.log(pred_te) + q)
cov = interval_coverage(y_te, lower_te, upper_te)
print("TEST interval coverage:", cov)

# --- Step 5: score test set once, vs baselines on the SAME rows
dummy = DummyRegressor(strategy="median").fit(X_tr, y_tr)
pred_dummy = dummy.predict(X_te)

class NeighborhoodSqftRule:
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        ratio = pd.Series(np.asarray(y), index=X.index) / X["sqft"].replace(0, np.nan)
        self.by_hood_ = ratio.groupby(X["neighborhood"]).median()
        self.overall_ = ratio.median()
        return self
    def predict(self, X):
        X = pd.DataFrame(X)
        r = X["neighborhood"].map(self.by_hood_).fillna(self.overall_)
        return (r * X["sqft"].fillna(X["sqft"].median())).to_numpy()

rule = NeighborhoodSqftRule().fit(X_tr, y_tr)
pred_rule = rule.predict(X_te)

def mae(y, p): return mean_absolute_error(y, p)
def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mape(y, p): return float(np.mean(np.abs((np.asarray(y) - np.asarray(p)) / np.asarray(y))) * 100)

results = {}
for name, pred in [("trivial_median", pred_dummy), ("business_rule", pred_rule), ("final_ridge_pipeline", pred_te)]:
    row = {}
    for metric_name, fn in [("MAE", mae), ("RMSE", rmse), ("MAPE_pct", mape)]:
        point, lo, hi = bootstrap_ci(y_te, pred, fn, n_boot=2000, stratified=False)
        row[metric_name] = {"point": round(point, 1), "ci95": [round(lo, 1), round(hi, 1)]}
    results[name] = row
    print(name, row)

json.dump({"test_metrics": results, "interval_coverage_test": cov,
           "conformal": {"method": "split-conformal, hand-implemented (MAPIE not installed)",
                         "nonconformity": "abs(log(y) - log(pred))", "calibration_set": "holdout (n=649)",
                         "target_coverage": 1 - ALPHA, "q_log_space": q}},
          open(f"{OUT}/test_metrics.json", "w"), indent=2)

# --- Step 6: slices
slices = pd.DataFrame({
    "neighborhood": test_df["neighborhood"],
    "condition": test_df["condition"].astype(str),
    "price_decile": pd.qcut(test_df["price"].rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)]),
    "sold_quarter": test_df["sold_date"].dt.to_period("Q").astype(str),
})
stab = slice_table(slices, y_te, pred_te, mae, min_n=15)
stab.to_csv(f"{OUT}/slice_table.csv", index=False)
print("\nworst slices by MAE:\n", stab.sort_values("metric", ascending=False).head(8))

# --- Step 7: error analysis
err = test_df[["listing_id", "neighborhood", "condition", "sqft", "price"]].copy()
err["pred"] = pred_te
err["abs_error"] = np.abs(err["price"] - err["pred"])
err["pct_error"] = err["abs_error"] / err["price"] * 100
err = err.sort_values("abs_error", ascending=False)
err.head(20).to_csv(f"{OUT}/worst_errors.csv", index=False)

# --- Step 8: sanity tests
# 8a: shuffled labels on train folds
def scorer(pipe, X, y):
    return -mean_absolute_error(y, pipe.predict(X))
real, fake = shuffled_label_check(build_final(), train[FEATS], train["price"].to_numpy(), s.folds, scorer)
print(f"\nshuffled-label check: real={-real:.1f} MAE, shuffled={-fake:.1f} MAE "
      f"(trivial baseline ~{mean_absolute_error(y_tr, np.full_like(y_tr, np.median(y_tr), dtype=float)):.1f})")

# 8b: duplicate row -> identical prediction
dup = X_te.iloc[[0, 0]]
p_dup = final_pipe.predict(dup)
print("duplicate-row check:", "PASS" if p_dup[0] == p_dup[1] else "FAIL", p_dup)

# 8c: raw uncleaned rows through the FULL production pipeline (clean_raw + final_pipe)
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import OUT as _  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location("stage1", f"{OUT}/code/01_stage1_audit.py")
stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1)

raw = pd.read_csv(r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv")
sample_raw = raw.sample(5, random_state=0).copy()
sample_raw.loc[sample_raw.index[0], "garage"] = "YES"      # spelling variant
sample_raw.loc[sample_raw.index[1], "sqft"] = 100000        # sentinel
sample_raw.loc[sample_raw.index[2], "lot_size_sqft"] = np.nan  # already-missing
cleaned = stage1.clean_raw(sample_raw)
cleaned["sold_year"] = cleaned["sold_date"].dt.year
cleaned["sold_month"] = cleaned["sold_date"].dt.month
try:
    p_raw = final_pipe.predict(cleaned[FEATS])
    print("raw-row-through-pipeline check: PASS ->", np.round(p_raw, 0).tolist())
except Exception as e:
    print("raw-row-through-pipeline check: FAIL ->", e)

# 8d: directional check - increasing sqft should raise the prediction
base_row = X_te.iloc[[0]].copy()
up_row = base_row.copy()
up_row["sqft"] = up_row["sqft"] + 500
p_base, p_up = final_pipe.predict(base_row)[0], final_pipe.predict(up_row)[0]
print(f"directional check (sqft +500): {p_base:.0f} -> {p_up:.0f}", "PASS" if p_up > p_base else "FAIL")

# --- Step 9: explanations - permutation importance (on holdout, held-out from training)
r = permutation_importance(final_pipe, X_ho, y_ho, scoring="neg_mean_absolute_error", n_repeats=15, random_state=0)
imp = pd.DataFrame({"feature": FEATS, "importance_mae_increase": r.importances_mean, "sd": r.importances_std}).sort_values("importance_mae_increase", ascending=False)
imp.to_csv(f"{OUT}/permutation_importance.csv", index=False)
print("\ntop permutation importances:\n", imp.head(8))

print("\nAll Stage 9 artifacts written to", OUT)
