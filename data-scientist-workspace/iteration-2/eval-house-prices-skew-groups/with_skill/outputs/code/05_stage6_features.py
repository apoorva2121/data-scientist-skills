"""Stage 6: feature engineering. Ledger + two screens: (a) engineered-feature-group ablation
(house_age, total_rooms, sold_year/month) vs base columns, both proxies; (b) neighborhood encoding,
one-hot vs cross-fitted TargetEncoder, linear proxy only (GBDT already uses native categorical
handling for neighborhood, so encoding choice does not apply to it).
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import Ridge

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, make_gbdt_pipe, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
from outlier_transformers import Winsorizer  # noqa
from screen import evaluate, summarize

train, s = load_train()
folds = s.folds
NOISE = 289.57


class AddDomainFeatures(BaseEstimator, TransformerMixin):
    """Stateless (fitted-in-fold-safe, no data-derived state) feature construction from raw
    availability-at-listing columns: house_age, total_rooms, sold_year, sold_month."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["house_age"] = X["sold_year"] - X["year_built"]
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        return X


def prep_frame(df):
    df = df.copy()
    df["sold_year"] = df["sold_date"].dt.year
    df["sold_month"] = df["sold_date"].dt.month
    return df


train2 = prep_frame(train)
y = train2["price"].to_numpy()

BASE_NUM = NUM_COLS
ENG_NUM = NUM_COLS + ["house_age", "total_rooms", "sold_year", "sold_month"]

def linear_pipe(num_cols, cat_cols, encoding="onehot"):
    cat_step = (OneHotEncoder(handle_unknown="ignore") if encoding == "onehot"
                else TargetEncoder(target_type="continuous", random_state=0))
    pre = ColumnTransformer([
        ("num", Pipeline([("wins", ColumnTransformer(
                              [("w", Winsorizer(0.01, 0.99), [num_cols.index(c) for c in ["lot_size_sqft", "distance_to_center_km"] if c in num_cols])],
                              remainder="passthrough")),
                           ("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("enc", cat_step)]), cat_cols),
    ])
    steps = []
    if len(num_cols) > len(NUM_COLS):
        steps.append(("feat", AddDomainFeatures()))
    steps += [("pre", pre), ("model", TransformedTargetRegressor(Ridge(alpha=1.0), func=np.log, inverse_func=np.exp))]
    return Pipeline(steps)


def gbdt_pipe_eng(num_cols, cat_cols):
    p = make_gbdt_pipe(num_cols=num_cols, cat_cols=cat_cols, impute="median")
    if len(num_cols) > len(NUM_COLS):
        return Pipeline([("feat", AddDomainFeatures())] + list(p.steps))
    return p


X_lin_base = train2[BASE_NUM + CAT_COLS]
X_lin_eng = train2[["sold_year", "sold_month"] + BASE_NUM + CAT_COLS]  # sold_year/month present pre-feat

# --- (a) engineered feature-group ablation
lin_candidates = {
    "base_cols": linear_pipe(BASE_NUM, CAT_COLS),
    "plus_engineered": linear_pipe(ENG_NUM, CAT_COLS),
}
print("=== linear: feature-group ablation ===")
# evaluate() takes one X for all candidates; build a combined superset frame so both pipelines
# can select their own columns via the ColumnTransformer's column list.
X_super = train2[["sold_year", "sold_month"] + BASE_NUM + CAT_COLS]
long_a = evaluate(lin_candidates, X_super, y, folds=folds, scoring="neg_mean_absolute_error")
tab_a = summarize(long_a, noise_floor_sd=NOISE)
print(tab_a)
tab_a.to_csv(f"{OUT}/stage6_feature_ablation_linear.csv", index=False)

gbdt_candidates = {
    "base_cols": make_gbdt_pipe(num_cols=BASE_NUM, cat_cols=CAT_COLS, impute="median"),
    "plus_engineered": gbdt_pipe_eng(ENG_NUM, CAT_COLS),
}
print("\n=== gbdt: feature-group ablation ===")
long_b = evaluate(gbdt_candidates, X_super, y, folds=folds, scoring="neg_mean_absolute_error")
tab_b = summarize(long_b, noise_floor_sd=NOISE)
print(tab_b)
tab_b.to_csv(f"{OUT}/stage6_feature_ablation_gbdt.csv", index=False)

# --- (b) neighborhood encoding, linear only
enc_candidates = {
    "onehot": linear_pipe(ENG_NUM, CAT_COLS, encoding="onehot"),
    "target_encoding": linear_pipe(ENG_NUM, CAT_COLS, encoding="target"),
}
print("\n=== linear: neighborhood encoding ===")
long_c = evaluate(enc_candidates, X_super, y, folds=folds, scoring="neg_mean_absolute_error")
tab_c = summarize(long_c, noise_floor_sd=NOISE)
print(tab_c)
tab_c.to_csv(f"{OUT}/stage6_encoding_screen_linear.csv", index=False)

# --- feature ledger
ledger = pd.DataFrame([
    {"feature": "sqft", "group": "base", "inputs": "sqft", "transformation": "none",
     "available_at_prediction": "yes: property record, known before listing", "fitted_in_fold": "n/a", "families": "all"},
    {"feature": "bedrooms/bathrooms", "group": "base", "inputs": "bedrooms,bathrooms", "transformation": "none",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "n/a", "families": "all"},
    {"feature": "year_built", "group": "base", "inputs": "year_built", "transformation": "none",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "n/a", "families": "all"},
    {"feature": "lot_size_sqft", "group": "base", "inputs": "lot_size_sqft", "transformation": "winsorize p01/p99 (linear only)",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "yes (Winsorizer)", "families": "all"},
    {"feature": "neighborhood", "group": "base", "inputs": "neighborhood", "transformation": "one-hot",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "yes (OneHotEncoder)", "families": "all"},
    {"feature": "distance_to_center_km", "group": "base", "inputs": "distance_to_center_km", "transformation": "winsorize p01/p99 (linear only)",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "yes (Winsorizer)", "families": "all"},
    {"feature": "condition", "group": "base", "inputs": "condition", "transformation": "one-hot (ordinal-aware category order kept)",
     "available_at_prediction": "yes (ASSUMED: set at listing/inspection)", "fitted_in_fold": "yes", "families": "all"},
    {"feature": "garage", "group": "base", "inputs": "garage", "transformation": "one-hot",
     "available_at_prediction": "yes: property record", "fitted_in_fold": "yes", "families": "all"},
    {"feature": "tax_assessed_value", "group": "base", "inputs": "tax_assessed_value", "transformation": "none",
     "available_at_prediction": "yes: public tax record, assessed before listing", "fitted_in_fold": "n/a", "families": "all"},
    {"feature": "house_age", "group": "engineered", "inputs": "sold_date,year_built", "transformation": "sold_year - year_built",
     "available_at_prediction": "yes: both inputs known before/at listing (sold_date = listing/close date)", "fitted_in_fold": "no (stateless)", "families": "all"},
    {"feature": "total_rooms", "group": "engineered", "inputs": "bedrooms,bathrooms", "transformation": "sum",
     "available_at_prediction": "yes", "fitted_in_fold": "no (stateless)", "families": "all"},
    {"feature": "sold_year/sold_month", "group": "engineered", "inputs": "sold_date", "transformation": "calendar parts",
     "available_at_prediction": "yes: the listing tool sets the intended sale/valuation date", "fitted_in_fold": "no (stateless)", "families": "all"},
    {"feature": "price_per_sqft", "group": "dropped", "inputs": "price,sqft", "transformation": "ratio",
     "available_at_prediction": "NO - post-sale only", "fitted_in_fold": "n/a", "families": "none (dropped Stage 3)"},
])
ledger.to_csv(f"{OUT}/feature_ledger.csv", index=False)

path = f"{OUT}/search_space.json"
space = json.load(open(path))
def rows(tab, fam): return [{"name": r["name"], "applies_to": [fam],
                              "proxy_scores": {fam: [round(r["mean"], 2), round(r["sd"], 2)]},
                              "keep": bool(r["keep"])} for _, r in tab.iterrows()]
space["stages"]["features"] = rows(tab_a, "linear") + rows(tab_b, "gbdt")
space["stages"]["encoding"] = rows(tab_c, "linear")
json.dump(space, open(path, "w"), indent=2)
print("wrote", path)
