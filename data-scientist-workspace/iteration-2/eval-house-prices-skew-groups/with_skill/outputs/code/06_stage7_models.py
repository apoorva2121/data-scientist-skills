"""Stage 7: baselines + model family screen + learnability verdict.

Data size (4320 rows) puts this in ds-model-selection's "up to ~5,000 rows" band, whose
recommended families are TabPFN-3.5 / TabICLv2, CatBoost, regularised linear. TabPFN/TabICL are
NOT installed and installing is not permitted in this environment -> logged as skipped with the
install note, listed as a report next-step. Screened instead: DummyRegressor baseline, a
comps business rule, Ridge (Stage 4-6 winning preprocessing), a depth-3 tree, LightGBM, CatBoost.
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import learning_curve
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, FrameImputer, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
from outlier_transformers import Winsorizer
from screen import evaluate, summarize

train, s = load_train()
folds = s.folds
NOISE = 289.57

train = train.copy()
train["sold_year"] = train["sold_date"].dt.year
train["sold_month"] = train["sold_date"].dt.month
ENG_NUM = NUM_COLS + ["house_age", "total_rooms", "sold_year", "sold_month"]
X = train[["sold_year", "sold_month"] + NUM_COLS + CAT_COLS]
y = train["price"].to_numpy()


class AddDomainFeatures(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["house_age"] = X["sold_year"] - X["year_built"]
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        return X


class NeighborhoodSqftRule(BaseEstimator, RegressorMixin):
    """Business-rule baseline: predicted_price = median($/sqft | neighborhood) x sqft,
    with an overall-median fallback for unseen/rare neighborhoods. Fitted in-fold."""
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


def linear_final():
    pre = ColumnTransformer([
        ("num", Pipeline([("wins", ColumnTransformer(
                              [("w", Winsorizer(0.01, 0.99), [ENG_NUM.index(c) for c in ["lot_size_sqft", "distance_to_center_km"]])],
                              remainder="passthrough")),
                           ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ENG_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("feat", AddDomainFeatures()), ("pre", pre),
                      ("model", TransformedTargetRegressor(Ridge(alpha=1.0), func=np.log, inverse_func=np.exp))])


def tree_shallow():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ENG_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("feat", AddDomainFeatures()), ("pre", pre),
                      ("model", TransformedTargetRegressor(DecisionTreeRegressor(max_depth=3, random_state=0), func=np.log, inverse_func=np.exp))])


def lgbm_final():
    imp = FrameImputer(ENG_NUM, CAT_COLS, add_indicator=False)
    model = TransformedTargetRegressor(
        LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, random_state=0, verbosity=-1),
        func=np.log, inverse_func=np.exp)
    return Pipeline([("feat", AddDomainFeatures()), ("prep", imp), ("model", model)])


def catboost_final():
    imp = FrameImputer(ENG_NUM, CAT_COLS, add_indicator=False)
    model = TransformedTargetRegressor(
        CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, cat_features=CAT_COLS,
                           random_state=0, verbose=False), func=np.log, inverse_func=np.exp)
    return Pipeline([("feat", AddDomainFeatures()), ("prep", imp), ("model", model)])


candidates = {
    "trivial_median": Pipeline([("model", DummyRegressor(strategy="median"))]),
    "business_rule_sqft_neighborhood": Pipeline([("model", NeighborhoodSqftRule())]),
    "ridge_final_prep": linear_final(),
    "tree_depth3": tree_shallow(),
    "lightgbm_final_prep": lgbm_final(),
    # catboost_final_prep: attempted, dropped - CatBoostRegressor(cat_features=[...]) breaks
    # sklearn.base.clone() (RuntimeError: constructor does not roundtrip cat_features), which
    # screen.py's evaluate() requires; confirmed by running it, not assumed. See decision log.
}

print("=== Stage 7: baselines + family screen (neg MAE $, Stage-2 folds) ===")
long_ = evaluate(candidates, X, y, folds=folds, scoring="neg_mean_absolute_error")
tab = summarize(long_, noise_floor_sd=NOISE)
print(tab)
tab.to_csv(f"{OUT}/stage7_baseline_family_screen.csv", index=False)

# learning curve for the best family (gbdt-ish) vs Ridge
best_name = tab.iloc[0]["name"]
print(f"\nlearning curve for {best_name} and ridge_final_prep:")
for name in [best_name, "ridge_final_prep"]:
    pipe = candidates[name]
    try:
        sizes, tr_sc, va_sc = learning_curve(pipe, X, y, cv=folds, scoring="neg_mean_absolute_error",
                                              train_sizes=[0.25, 0.5, 0.75, 1.0])
        print(name, "val MAE by size:", [round(-v, 0) for v in va_sc.mean(axis=1)])
    except Exception as e:
        print(name, "learning_curve failed:", e)

path = f"{OUT}/search_space.json"
space = json.load(open(path))
space["stages"]["models"] = [{"name": r["name"], "mean": r["mean"], "sd": r["sd"], "keep": bool(r["keep"])}
                              for _, r in tab.iterrows()]
json.dump(space, open(path, "w"), indent=2)
print("wrote", path)
