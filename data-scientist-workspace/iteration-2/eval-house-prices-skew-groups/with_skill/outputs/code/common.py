"""Shared loaders and the two Stage 4-7 proxy-pipeline factories (linear = Ridge, gbdt = LightGBM),
both modeling log(price) internally via TransformedTargetRegressor so every scorer works in raw
dollars (MAE/RMSE), matching the primary metric in problem_statement.md.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs"
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-outliers\scripts")
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
from cv_folds import load_split  # noqa: E402
from outlier_transformers import Winsorizer  # noqa: E402

NUM_COLS = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
            "distance_to_center_km", "tax_assessed_value"]
CAT_COLS = ["neighborhood", "condition", "garage"]
TAIL_COLS = ["lot_size_sqft", "distance_to_center_km"]  # heavy-tail, Stage 5 verdict


def load_train():
    df = pd.read_csv(f"{OUT}/house_prices_stage1_clean.csv", parse_dates=["sold_date"])
    df["listing_id"] = df["listing_id"].astype(str)
    df["property_id"] = df["property_id"].astype(str)
    s = load_split(f"{OUT}/split/split.csv", ids=df["listing_id"])
    train = df[s.train_mask].reset_index(drop=True)
    return train, s


class FrameImputer(BaseEstimator, TransformerMixin):
    """Median-impute numeric cols (train-fold medians only), fill categoricals with a 'Missing'
    level, optional missing-indicator columns. Returns a DataFrame with category dtype so
    LightGBM's native categorical handling still applies after imputation."""

    def __init__(self, num_cols, cat_cols, add_indicator=False):
        self.num_cols, self.cat_cols, self.add_indicator = num_cols, cat_cols, add_indicator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.medians_ = {c: X[c].median() for c in self.num_cols}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.add_indicator:
            for c in self.num_cols:
                X[f"{c}__missing"] = X[c].isna().astype(int)
        for c in self.num_cols:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.cat_cols:
            X[c] = X[c].astype("object").fillna("Missing").astype("category")
        return X


class NativeNaNFrame(BaseEstimator, TransformerMixin):
    """No imputation: numeric NaNs pass through for LightGBM's native missing-split handling;
    categorical NaNs become their own 'category' level (LightGBM bins them natively too)."""

    def __init__(self, num_cols, cat_cols):
        self.num_cols, self.cat_cols = num_cols, cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_cols:
            X[c] = X[c].astype("category")
        return X


def make_linear_pipe(num_cols=NUM_COLS, cat_cols=CAT_COLS, add_indicator=True,
                      outlier="none", alpha=1.0):
    num_steps = []
    if outlier == "winsorize":
        num_steps.append(("wins", ColumnTransformer(
            [("w", Winsorizer(0.01, 0.99), [num_cols.index(c) for c in TAIL_COLS if c in num_cols])],
            remainder="passthrough")))
    num_steps += [("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
                  ("scale", StandardScaler())]
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    model = TransformedTargetRegressor(Ridge(alpha=alpha), func=np.log, inverse_func=np.exp)
    return Pipeline([("pre", pre), ("model", model)])


def make_gbdt_pipe(num_cols=NUM_COLS, cat_cols=CAT_COLS, impute="native", **lgbm_kwargs):
    imputer = (FrameImputer(num_cols, cat_cols, add_indicator=False) if impute == "median"
               else NativeNaNFrame(num_cols, cat_cols))
    defaults = dict(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20,
                     random_state=0, verbosity=-1)
    defaults.update(lgbm_kwargs)
    model = TransformedTargetRegressor(LGBMRegressor(**defaults), func=np.log, inverse_func=np.exp)
    return Pipeline([("prep", imputer), ("model", model)])
