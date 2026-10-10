"""
Fit the production model on ALL available rows (the earlier train/
calibration/test split in 03_quantile_uncertainty.py already gave an
honest, leakage-free estimate of accuracy and interval coverage -- once
that number is in hand, standard practice is to refit on every row to get
the best possible model for deployment).

The conformal margins found in step 03 (fit on a calibration split that
was never used to train the quantile models) are carried over as-is for
scoring new listings. This is an approximation: strictly, CQR's coverage
guarantee holds for the model/margin pair it was validated with, not for
a subsequently-refit model. Because here the refit only adds ~30% more
rows of the same distribution, the margins should still be close to
correct, but this is noted as a caveat in report.md and periodic
re-calibration is recommended once real listing outcomes come in.

Outputs:
- final_model.joblib      -- dict: {"preprocessor", "quantile_models",
                              "conformal_margins", "quantiles"}
- example_new_listing_prediction.txt -- worked example of scoring one
  new (unsold) listing end-to-end.
"""
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from data_prep import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_X_y_groups, load_and_clean

# Note: 03_quantile_uncertainty.py can't be imported as a module (filename
# starts with a digit), so the small helpers needed here are re-declared
# below instead, keeping this script standalone.

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def make_preprocessor():
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def group_three_way_split(X, y, groups, random_state=RANDOM_STATE):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=random_state)
    train_idx, rest_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    cal_rel_idx, test_rel_idx = next(gss2.split(X.iloc[rest_idx], y.iloc[rest_idx], groups.iloc[rest_idx]))
    cal_idx = rest_idx[cal_rel_idx]
    test_idx = rest_idx[test_rel_idx]
    return train_idx, cal_idx, test_idx


def cqr_margin_for_level(y_cal, lo_pred, hi_pred, nominal_coverage):
    n = len(y_cal)
    scores = np.maximum(lo_pred - y_cal, y_cal - hi_pred)
    alpha = 1 - nominal_coverage
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, level, method="higher")


def fit_quantile_models(prep, X_train, y_train_log):
    Xt = prep.fit_transform(X_train)
    models = {}
    for q in QUANTILES:
        m = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0,
        )
        m.fit(Xt, y_train_log)
        models[q] = m
    return models


def predict_quantiles(prep, models, X):
    Xt = prep.transform(X)
    preds_log = np.column_stack([models[q].predict(Xt) for q in QUANTILES])
    preds_log = np.sort(preds_log, axis=1)
    preds = np.expm1(preds_log)
    return pd.DataFrame(preds, columns=[f"q{int(q*100):02d}" for q in QUANTILES], index=X.index)


if __name__ == "__main__":
    df = load_and_clean()
    X, y, groups = get_X_y_groups(df)

    # Re-derive the same cal/test split as step 03 purely to recompute the
    # conformal margins (needed here since this script is standalone).
    train_idx, cal_idx, test_idx = group_three_way_split(X, y, groups)
    X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]

    # Margins from a model fit on the TRAIN split only (matches step 03).
    prep_val = make_preprocessor()
    models_val = fit_quantile_models(prep_val, X.iloc[train_idx], np.log1p(y.iloc[train_idx].values))
    q_cal = predict_quantiles(prep_val, models_val, X_cal)

    margins = {}
    for lo_q, hi_q, nominal in [(0.05, 0.95, 0.90), (0.10, 0.90, 0.80), (0.25, 0.75, 0.50)]:
        lo_col, hi_col = f"q{int(lo_q*100):02d}", f"q{int(hi_q*100):02d}"
        margins[f"{int(nominal*100)}"] = float(
            cqr_margin_for_level(y_cal.values, q_cal[lo_col].values, q_cal[hi_col].values, nominal)
        )
    print("Conformal margins carried into production model:", margins)

    # ---- Refit quantile models on ALL rows for deployment ----
    prep_final = make_preprocessor()
    models_final = fit_quantile_models(prep_final, X, np.log1p(y.values))

    bundle = {
        "preprocessor": prep_final,
        "quantile_models": models_final,
        "quantiles": QUANTILES,
        "conformal_margins": margins,  # keyed by nominal coverage % using q10/q90 (80%), q05/q95(90%), q25/q75(50%)
        "features": ALL_FEATURES,
    }
    joblib.dump(bundle, "final_model.joblib")
    print("Saved final_model.joblib")

    # ---- Worked example: score one new, unsold listing ----
    example = pd.DataFrame([{
        "sqft": 1850, "bedrooms": 3, "bathrooms": 2, "year_built": 1998,
        "lot_size_sqft": 6200, "distance_to_center_km": 6.5,
        "tax_assessed_value": 105000, "house_age": 2026 - 1998,
        "sale_year": 2026, "sale_month": 6,
        "sqft_outlier_flag": 0, "year_built_outlier_flag": 0,
        "lot_size_missing": 0, "garage_missing": 0,
        "neighborhood": "N02", "condition": "Good", "garage_clean": "Yes",
    }])[ALL_FEATURES]

    q_pred = predict_quantiles(prep_final, models_final, example).iloc[0]
    point = q_pred["q50"]
    lo80 = max(q_pred["q10"] - margins["80"], 0)
    hi80 = q_pred["q90"] + margins["80"]

    lines = [
        "Example new-listing scoring (illustrative, not a real property):",
        str(example.iloc[0].to_dict()),
        f"Point estimate (median): ${point:,.0f}",
        f"80% interval (conformal-calibrated): ${lo80:,.0f} - ${hi80:,.0f}",
    ]
    print("\n".join(lines))
    with open("example_new_listing_prediction.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
