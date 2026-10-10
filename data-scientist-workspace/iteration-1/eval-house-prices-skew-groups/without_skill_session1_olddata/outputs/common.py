"""
common.py - shared loading, cleaning, feature engineering, CV splitting and metrics.
Imported by the modelling scripts (03_, 04_, 05_, 06_).
"""
import os
# Single-threaded BLAS/OpenMP: on this (heavily loaded, Windows) machine multi-threaded LightGBM/XGBoost/HistGBR
# were 5-10x SLOWER than single-threaded due to thread contention (measured: LGBM 3.1s @1 thread vs 30.4s @4).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
N_THREADS = 1

DATA = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"
SEED = 42
COND_ORDER = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4}
T0 = pd.Timestamp("2021-01-01")

NUM_FEATURES = [
    "sqft", "bedrooms", "bathrooms", "year_built", "age_at_sale", "lot_size_sqft",
    "lot_size_missing", "distance_to_center_km", "condition_ord", "garage_bin",
    "tax_assessed_value", "log_tax", "tax_per_sqft", "sold_year", "sold_month", "t_months",
]
CAT_FEATURES = ["neighborhood"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["sold_date"] = pd.to_datetime(df["sold_date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise cleaning. No fitting => safe to apply identically at inference."""
    df = df.copy()
    # sentinel / impossible values -> NaN (tree models handle NaN natively; linear models impute)
    df.loc[(df["sqft"] < 300) | (df["sqft"] > 10000), "sqft"] = np.nan
    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = np.nan
    # garage label normalisation (Yes/yes/YES/Y -> 1, No/no/N -> 0, missing -> NaN)
    g = df["garage"].astype("string").str.strip().str.lower()
    df["garage_bin"] = g.map({"yes": 1.0, "y": 1.0, "no": 0.0, "n": 0.0}).astype(float)
    # ordinal condition
    df["condition_ord"] = df["condition"].map(COND_ORDER).astype(float)
    # lot size missing flag
    df["lot_size_missing"] = df["lot_size_sqft"].isna().astype(float)
    # time features (for a new listing, sold_date = expected sale / listing date)
    df["sold_year"] = df["sold_date"].dt.year.astype(float)
    df["sold_month"] = df["sold_date"].dt.month.astype(float)
    df["t_months"] = ((df["sold_date"] - T0).dt.days / 30.44).astype(float)
    df["age_at_sale"] = df["sold_year"] - df["year_built"]
    # tax-derived
    df["log_tax"] = np.log(df["tax_assessed_value"])
    df["tax_per_sqft"] = df["tax_assessed_value"] / df["sqft"]
    # categorical
    df["neighborhood"] = df["neighborhood"].astype("category")
    # drop leakage column (it is literally price / sqft)
    if "price_per_sqft" in df.columns:
        df = df.drop(columns=["price_per_sqft"])
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def get_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = np.log(df["price"].values)
    groups = df["property_id"].values
    return X, y, groups


def group_cv(n_splits=5, seed=SEED):
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def random_cv(n_splits=5, seed=SEED):
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def metrics(y_log_true, y_log_pred) -> dict:
    """Metrics in log space and in dollar space (point estimate = exp(pred) = conditional median)."""
    y_true = np.exp(y_log_true)
    y_pred = np.exp(y_log_pred)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "rmse_log": float(np.sqrt(np.mean((y_log_pred - y_log_true) ** 2))),
        "mae_log": float(np.mean(np.abs(y_log_pred - y_log_true))),
        "rmse_usd": float(np.sqrt(np.mean(err ** 2))),
        "mae_usd": float(np.mean(np.abs(err))),
        "mape_pct": float(100 * np.mean(ape)),
        "mdape_pct": float(100 * np.median(ape)),
        "within_10pct": float(100 * np.mean(ape <= 0.10)),
        "r2_usd": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def fmt(m: dict) -> str:
    return (f"rmse_log={m['rmse_log']:.4f} mae_log={m['mae_log']:.4f} | "
            f"MAE=${m['mae_usd']:,.0f} RMSE=${m['rmse_usd']:,.0f} MAPE={m['mape_pct']:.2f}% "
            f"MdAPE={m['mdape_pct']:.2f}% within10%={m['within_10pct']:.1f}% R2={m['r2_usd']:.3f}")
