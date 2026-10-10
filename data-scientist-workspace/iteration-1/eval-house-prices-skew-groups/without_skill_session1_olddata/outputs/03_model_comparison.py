"""
03_model_comparison.py - Compare candidate models with 5-fold GroupKFold (grouped by property_id),
show the leakage effect of ungrouped KFold, run a temporal holdout sanity check, and an ablation
without tax_assessed_value.
All models predict log(price). Metrics reported in both log and dollar space.
"""
from common import N_THREADS  # sets OMP/MKL thread env vars before sklearn/lightgbm import
import time, json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from common import (load_clean, get_xy, group_cv, random_cv, metrics, fmt,
                    NUM_FEATURES, CAT_FEATURES, FEATURES, SEED)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

df = load_clean()
X, y, groups = get_xy(df)
print("rows:", len(df), "features:", len(FEATURES))
print("NaN counts after cleaning:\n", X.isna().sum()[X.isna().sum() > 0])


# ---------- model factories ----------
def make_linear_pre(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def make_tree_pre(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


class TaxRatioBaseline:
    """price_hat = tax_assessed_value * median(price/tax) on train. In log space: log_tax + c."""
    def fit(self, X, y):
        self.c_ = np.median(y - X["log_tax"].values); return self
    def predict(self, X):
        return X["log_tax"].values + self.c_


class MedianBaseline:
    def fit(self, X, y):
        self.m_ = np.median(y); return self
    def predict(self, X):
        return np.full(len(X), self.m_)


class LGBWrapper:
    def __init__(self, **params):
        self.params = dict(objective="regression", learning_rate=0.03, n_estimators=2000,
                           num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, num_threads=N_THREADS)
        self.params.update(params)
    def fit(self, X, y, X_val=None, y_val=None):
        self.m_ = lgb.LGBMRegressor(**self.params)
        cats = [c for c in CAT_FEATURES if c in X.columns]
        if X_val is not None:
            self.m_.fit(X, y, eval_set=[(X_val, y_val)], categorical_feature=cats,
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            self.m_.fit(X, y, categorical_feature=cats)
        return self
    def predict(self, X):
        return self.m_.predict(X)


class XGBWrapper:
    def __init__(self, **params):
        self.params = dict(objective="reg:squarederror", learning_rate=0.03, n_estimators=2000,
                           max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, enable_categorical=True, tree_method="hist",
                           random_state=SEED, early_stopping_rounds=100, n_jobs=N_THREADS)
        self.params.update(params)
    def fit(self, X, y, X_val=None, y_val=None):
        p = dict(self.params)
        if X_val is None:
            p.pop("early_stopping_rounds")
        self.m_ = xgb.XGBRegressor(**p)
        if X_val is not None:
            self.m_.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.m_.fit(X, y)
        return self
    def predict(self, X):
        return self.m_.predict(X)


def build_models(num_cols=NUM_FEATURES, cat_cols=CAT_FEATURES):
    return {
        "baseline_median": lambda: MedianBaseline(),
        "baseline_tax_ratio": lambda: TaxRatioBaseline(),
        "ridge_onehot": lambda: Pipeline([("pre", make_linear_pre(num_cols, cat_cols)), ("m", Ridge(alpha=1.0))]),
        "random_forest": lambda: Pipeline([("pre", make_tree_pre(num_cols, cat_cols)),
                                           ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                                       max_features=0.5, n_jobs=4, random_state=SEED))]),
        "hist_gbr": lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, max_leaf_nodes=31,
                                                          min_samples_leaf=20, l2_regularization=1.0,
                                                          categorical_features="from_dtype",
                                                          early_stopping=True, validation_fraction=0.15,
                                                          random_state=SEED),
        "lightgbm": lambda: LGBWrapper(),
        "xgboost": lambda: XGBWrapper(),
    }


def run_cv(model_fn, X, y, groups, splitter, use_es=False):
    """Return OOF predictions and per-fold metrics."""
    oof = np.zeros(len(y))
    fold_m = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        m = model_fn()
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        if use_es:
            # inner early-stopping split, grouped, so the fold's test rows are never used for stopping
            inner = group_cv(5, seed=SEED + k)
            itr, iva = next(inner.split(Xtr, y[tr], groups[tr]))
            m.fit(Xtr.iloc[itr], y[tr][itr], Xtr.iloc[iva], y[tr][iva])
        else:
            m.fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
        fold_m.append(metrics(y[te], oof[te]))
    return oof, pd.DataFrame(fold_m)


results = {}
oofs = {}
models = build_models()
print("\n================ 5-fold GroupKFold by property_id (primary) ================")
for name, fn in models.items():
    t = time.time()
    oof, fm = run_cv(fn, X, y, groups, group_cv(), use_es=name in ("lightgbm", "xgboost"))
    m = metrics(y, oof)
    results[name] = {"overall": m,
                     "fold_rmse_log_mean": fm["rmse_log"].mean(), "fold_rmse_log_std": fm["rmse_log"].std(),
                     "fold_mae_usd_mean": fm["mae_usd"].mean(), "fold_mae_usd_std": fm["mae_usd"].std()}
    oofs[name] = oof
    print(f"{name:20s} {fmt(m)} | fold rmse_log {fm['rmse_log'].mean():.4f}+/-{fm['rmse_log'].std():.4f}  ({time.time()-t:.1f}s)")

# simple blends
blend = (oofs["lightgbm"] + oofs["xgboost"]) / 2
blend3 = (oofs["lightgbm"] + oofs["xgboost"] + oofs["ridge_onehot"]) / 3
print(f"{'blend_lgb_xgb':20s} {fmt(metrics(y, blend))}")
print(f"{'blend_lgb_xgb_ridge':20s} {fmt(metrics(y, blend3))}")
results["blend_lgb_xgb"] = {"overall": metrics(y, blend)}
results["blend_lgb_xgb_ridge"] = {"overall": metrics(y, blend3)}

# paired fold comparison (noise floor)
print("\n--- Per-fold paired comparison (rmse_log), same GroupKFold splits ---")
splits = list(group_cv().split(X, y, groups))
rows = []
for k, (tr, te) in enumerate(splits):
    r = {n: metrics(y[te], oofs[n][te])["rmse_log"] for n in ["ridge_onehot", "random_forest", "hist_gbr", "lightgbm", "xgboost"]}
    r["blend"] = metrics(y[te], blend[te])["rmse_log"]
    rows.append(r)
pf = pd.DataFrame(rows); print(pf.round(4)); print("std across folds:", pf.std().round(4).to_dict())

print("\n================ Leakage check: ungrouped KFold on the same models ================")
rep_mask = df["property_id"].duplicated(keep=False).values
for name in ["random_forest", "lightgbm"]:
    oof_r, fm_r = run_cv(models[name], X, y, groups, random_cv(), use_es=name == "lightgbm")
    m_all = metrics(y, oof_r)
    m_rep = metrics(y[rep_mask], oof_r[rep_mask]); m_rep_g = metrics(y[rep_mask], oofs[name][rep_mask])
    print(f"{name:14s} random KFold : {fmt(m_all)}")
    print(f"{'':14s}   on the 640 repeat-sale rows -> random KFold rmse_log={m_rep['rmse_log']:.4f} vs GroupKFold rmse_log={m_rep_g['rmse_log']:.4f}")
    results[name]["random_kfold_rmse_log"] = m_all["rmse_log"]

print("\n================ Temporal holdout sanity check (train < 2024-01-01, test >= 2024) ================")
cut = pd.Timestamp("2024-01-01")
tr = (df["sold_date"] < cut).values
te = ~tr
seen = df["property_id"].isin(df.loc[tr, "property_id"]).values
te_new = te & ~seen
print(f"train rows {tr.sum()}, test rows {te.sum()} (of which {te_new.sum()} are properties never seen in train)")
for name in ["baseline_tax_ratio", "ridge_onehot", "lightgbm", "xgboost"]:
    m = models[name]()
    if name in ("lightgbm", "xgboost"):
        inner = group_cv(5); itr, iva = next(inner.split(X[tr], y[tr], groups[tr]))
        m.fit(X[tr].iloc[itr], y[tr][itr], X[tr].iloc[iva], y[tr][iva])
    else:
        m.fit(X[tr], y[tr])
    p = m.predict(X[te]); p_new = m.predict(X[te_new])
    mt = metrics(y[te], p); mn = metrics(y[te_new], p_new)
    print(f"{name:20s} all-2024+: {fmt(mt)}")
    print(f"{'':20s} unseen-props only: rmse_log={mn['rmse_log']:.4f} MAE=${mn['mae_usd']:,.0f} MdAPE={mn['mdape_pct']:.2f}%")
    results[name]["temporal_holdout"] = mt
    results[name]["temporal_holdout_unseen"] = mn

print("\n================ Ablation: without tax_assessed_value (if not available for a listing) ================")
no_tax = [c for c in NUM_FEATURES if c not in ("tax_assessed_value", "log_tax", "tax_per_sqft")] + CAT_FEATURES
oof_nt, fm_nt = run_cv(lambda: LGBWrapper(), X[no_tax], y, groups, group_cv(), use_es=True)
print(f"{'lightgbm_no_tax':20s} {fmt(metrics(y, oof_nt))}")
results["lightgbm_no_tax"] = {"overall": metrics(y, oof_nt)}

print("\n================ LightGBM feature importance (gain %, fit on all data, 600 rounds) ================")
m = LGBWrapper(n_estimators=600).fit(X, y)
imp = pd.Series(m.m_.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print((imp / imp.sum() * 100).round(2))

# save
out = pd.DataFrame({"listing_id": df["listing_id"], "property_id": df["property_id"], "price": df["price"]})
for k, v in oofs.items():
    out[f"oof_{k}"] = np.exp(v)
out["oof_blend_lgb_xgb"] = np.exp(blend)
out.to_csv("03_oof_predictions.csv", index=False)
with open("03_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved 03_oof_predictions.csv, 03_results.json")
