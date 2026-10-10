"""The shipped production pipeline definition, importable so a saved skops file can be reloaded
and so the raw-row FunctionTransformer step has a real module path (required for serialization).

Pipeline: raw CSV row -> clean_raw (Stage 1, stateless) -> add sold_year/sold_month (stateless
calendar parts) -> AddDomainFeatures (house_age, total_rooms; stateless) -> preprocessing
(winsorize tails, median-impute, standardize numerics; most-frequent-impute + one-hot categoricals;
all fitted in-fold at training time) -> Ridge(alpha=0.011545) on log(price), back-transformed with
np.exp. Ships WITHOUT price_per_sqft (dropped, Stage 3: derived from the target).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Ridge

NUM_COLS = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
            "distance_to_center_km", "tax_assessed_value"]
CAT_COLS = ["neighborhood", "condition", "garage"]
ENG_NUM = NUM_COLS + ["house_age", "total_rooms", "sold_year", "sold_month"]
FEATS = ["sold_year", "sold_month"] + NUM_COLS + CAT_COLS
FINAL_ALPHA = 0.011545
CONFORMAL_Q_LOG = 0.2295  # 90%-target split-conformal margin, log space (Stage 9, calibrated on holdout n=649)

GARAGE_MAP = {"Yes": "Yes", "yes": "Yes", "Y": "Yes", "YES": "Yes", "No": "No", "no": "No", "N": "No"}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 stateless cleaning: sentinel->null, spelling normalization, type parsing.
    Identical to code/01_stage1_audit.py's clean_raw (duplicated here so this module is
    self-contained and importable independently for production loading)."""
    df = df.copy()
    df.loc[df["sqft"].isin([50, 100000]), "sqft"] = np.nan
    df.loc[df["lot_size_sqft"] == 9999, "lot_size_sqft"] = np.nan
    df.loc[df["year_built"].isin([1800, 2090]), "year_built"] = np.nan
    df["garage"] = df["garage"].map(GARAGE_MAP).where(df["garage"].notna(), np.nan)
    df["sold_date"] = pd.to_datetime(df["sold_date"], format="%Y-%m-%d")
    if "listing_id" in df.columns:
        df["listing_id"] = df["listing_id"].astype(str)
    if "property_id" in df.columns:
        df["property_id"] = df["property_id"].astype(str)
    df["condition"] = pd.Categorical(df["condition"], categories=["Poor", "Fair", "Good", "Very Good", "Excellent"], ordered=True)
    return df


def raw_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """FunctionTransformer body: raw listing row -> cleaned feature frame with calendar parts."""
    df = clean_raw(pd.DataFrame(df))
    df["sold_year"] = df["sold_date"].dt.year
    df["sold_month"] = df["sold_date"].dt.month
    return df[FEATS]


class AddDomainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["house_age"] = X["sold_year"] - X["year_built"]
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Duplicated from ds-outliers/scripts/outlier_transformers.py so this module has no external
    dependency at load time. Clip to [p01, p99] learned on the training fold."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_production_pipeline(alpha: float = FINAL_ALPHA) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("wins", ColumnTransformer(
                              [("w", Winsorizer(0.01, 0.99), [ENG_NUM.index(c) for c in ["lot_size_sqft", "distance_to_center_km"]])],
                              remainder="passthrough")),
                           ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ENG_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(raw_to_features)),
        ("feat", AddDomainFeatures()),
        ("pre", pre),
        ("model", TransformedTargetRegressor(Ridge(alpha=alpha), func=np.log, inverse_func=np.exp)),
    ])


def predict_with_interval(pipe: Pipeline, raw_df: pd.DataFrame, q_log: float = CONFORMAL_Q_LOG):
    """Point estimate + 90%-target split-conformal interval for raw, uncleaned rows."""
    pred = pipe.predict(raw_df)
    lower = np.exp(np.log(pred) - q_log)
    upper = np.exp(np.log(pred) + q_log)
    return pred, lower, upper
