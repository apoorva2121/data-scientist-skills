"""
Compare candidate point-estimate models with GROUP-aware 5-fold CV
(grouped by property_id, since 320 properties are sold twice and would
otherwise leak between train/validation folds).

Models compared:
- Linear Regression on log1p(price)            (simple baseline)
- Random Forest                                  (nonlinear baseline)
- LightGBM                                       (gradient boosting)
- XGBoost                                        (gradient boosting)
- CatBoost                                       (gradient boosting)

All tree models are fit on log1p(price) as well, since price is
right-skewed (skew ~1.6) and squared-error losses are dominated by the
handful of very expensive homes if fit on raw dollars; log-space fitting
+ expm1 inversion is a standard fix and also keeps errors closer to
percentage terms, which is the natural way to think about home price
error ("off by X%").

Metrics reported on the ORIGINAL dollar scale (after inverting the log
transform): RMSE, MAE, MAPE, R2. All are computed by concatenating
out-of-fold predictions across the 5 folds (not averaging per-fold
metrics), which is the more stable estimator for a dataset this size.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_X_y_groups, load_and_clean

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

RANDOM_STATE = 42
N_SPLITS = 5


def make_preprocessor(scale_numeric: bool):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def build_models():
    models = {}

    models["LinearRegression"] = Pipeline([
        ("prep", make_preprocessor(scale_numeric=True)),
        ("model", LinearRegression()),
    ])

    models["RandomForest"] = Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])

    models["LightGBM"] = Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=-1)),
    ])

    models["XGBoost"] = Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0)),
    ])

    models["CatBoost"] = Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", CatBoostRegressor(
            n_estimators=600, learning_rate=0.03, depth=6,
            random_state=RANDOM_STATE, verbose=False)),
    ])

    return models


def evaluate(models, X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    results = []
    for name, pipe in models.items():
        t0 = time.time()
        oof_pred = np.zeros(len(y))
        y_log = np.log1p(y.values)
        for tr_idx, va_idx in gkf.split(X, y, groups):
            pipe.fit(X.iloc[tr_idx], y_log[tr_idx])
            pred_log = pipe.predict(X.iloc[va_idx])
            oof_pred[va_idx] = np.expm1(pred_log)
        elapsed = time.time() - t0

        rmse = np.sqrt(mean_squared_error(y, oof_pred))
        mae = mean_absolute_error(y, oof_pred)
        mape = mean_absolute_percentage_error(y, oof_pred)
        r2 = r2_score(y, oof_pred)
        results.append({
            "model": name, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "R2": r2, "fit_seconds_total": elapsed,
        })
        print(f"{name:20s}  RMSE={rmse:9.0f}  MAE={mae:8.0f}  MAPE={mape:.4f}  R2={r2:.4f}  ({elapsed:.1f}s)")
    return pd.DataFrame(results).sort_values("RMSE")


if __name__ == "__main__":
    df = load_and_clean()
    X, y, groups = get_X_y_groups(df)

    print(f"n={len(df)}, n_groups(properties)={groups.nunique()}, n_features={len(ALL_FEATURES)}")
    print(f"GroupKFold, n_splits={N_SPLITS}\n")

    models = build_models()
    results_df = evaluate(models, X, y, groups)

    print("\n=== Summary (sorted by RMSE, out-of-fold, group-aware CV) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved model_comparison_results.csv")
