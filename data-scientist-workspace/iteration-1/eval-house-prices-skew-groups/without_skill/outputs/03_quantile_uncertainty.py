"""
Final model + uncertainty estimates.

Point-estimate model comparison (02_model_comparison.py, group-aware 5-fold
CV) put XGBoost narrowly ahead of LightGBM/CatBoost (RMSE ~17.7k vs ~17.9k-
18.0k; the gap is small relative to fold-to-fold variation but consistent
enough, and XGBoost natively supports multi-quantile training, so it is
used for both the point estimate and the uncertainty bands below).

Uncertainty method: Conformalized Quantile Regression (CQR).
  1. Split properties (not rows!) into train / calibration / test
     (60/20/20) with GroupShuffleSplit on property_id, so a property that
     was sold twice never appears in two different splits.
  2. Fit one XGBoost quantile-regression model per quantile level
     (objective="reg:quantileerror") on log1p(price), for quantiles
     {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}. Predictions are inverted
     with expm1 (valid because expm1 is monotonic, so quantiles commute
     with it) and sorted row-wise to remove any quantile crossing.
  3. On the untouched calibration split, compute the CQR nonconformity
     score for each candidate interval width and take the finite-sample-
     corrected empirical quantile of that score as an additive margin.
     This is what fixes the raw quantile model's miscalibration (gradient-
     boosted quantile regression is usually somewhat off nominal coverage
     on a dataset this size) -- CQR gives a *finite-sample marginal
     coverage guarantee* under the sole assumption of exchangeability
     between calibration and test points.
  4. Evaluate coverage/width on the held-out test split (never touched
     during fitting or conformal calibration) for the 50/80/90% intervals,
     both before and after conformal correction, overall and by price
     tercile (to check the calibration isn't just "correct on average").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb

from data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_X_y_groups, load_and_clean

RANDOM_STATE = 42
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
INTERVALS = [(0.05, 0.95, 0.90), (0.10, 0.90, 0.80), (0.25, 0.75, 0.50)]  # (lo, hi, nominal_coverage)


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
    # 60% train / 20% calibration / 20% test, split on property_id groups.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=random_state)
    train_idx, rest_idx = next(gss1.split(X, y, groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    cal_rel_idx, test_rel_idx = next(gss2.split(X.iloc[rest_idx], y.iloc[rest_idx], groups.iloc[rest_idx]))
    cal_idx = rest_idx[cal_rel_idx]
    test_idx = rest_idx[test_rel_idx]

    # sanity: no property_id shared across splits
    g_train = set(groups.iloc[train_idx])
    g_cal = set(groups.iloc[cal_idx])
    g_test = set(groups.iloc[test_idx])
    assert not (g_train & g_cal) and not (g_train & g_test) and not (g_cal & g_test)

    return train_idx, cal_idx, test_idx


def fit_quantile_models(prep, X_train, y_train_log):
    Xt = prep.fit_transform(X_train)
    models = {}
    for q in QUANTILES:
        m = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        m.fit(Xt, y_train_log)
        models[q] = m
    return models


def predict_quantiles(prep, models, X):
    Xt = prep.transform(X)
    preds_log = np.column_stack([models[q].predict(Xt) for q in QUANTILES])
    preds_log = np.sort(preds_log, axis=1)  # enforce monotonicity across quantiles
    preds = np.expm1(preds_log)
    return pd.DataFrame(preds, columns=[f"q{int(q*100):02d}" for q in QUANTILES], index=X.index)


def cqr_margin_for_level(y_cal, lo_pred, hi_pred, nominal_coverage):
    """Finite-sample-corrected CQR margin for a target nominal coverage."""
    n = len(y_cal)
    scores = np.maximum(lo_pred - y_cal, y_cal - hi_pred)
    alpha = 1 - nominal_coverage
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    margin = np.quantile(scores, level, method="higher")
    return margin


def coverage_and_width(y, lo, hi):
    covered = (y >= lo) & (y <= hi)
    return covered.mean(), (hi - lo).mean()


if __name__ == "__main__":
    df = load_and_clean()
    X, y, groups = get_X_y_groups(df)
    price = df["price"].reset_index(drop=True)

    train_idx, cal_idx, test_idx = group_three_way_split(X, y, groups)
    print(f"train n={len(train_idx)} ({groups.iloc[train_idx].nunique()} properties)")
    print(f"cal   n={len(cal_idx)} ({groups.iloc[cal_idx].nunique()} properties)")
    print(f"test  n={len(test_idx)} ({groups.iloc[test_idx].nunique()} properties)")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    prep = make_preprocessor()
    print("\nFitting 7 quantile models (XGBoost, reg:quantileerror)...")
    models = fit_quantile_models(prep, X_train, np.log1p(y_train.values))

    q_train = predict_quantiles(prep, models, X_train)
    q_cal = predict_quantiles(prep, models, X_cal)
    q_test = predict_quantiles(prep, models, X_test)

    # ---- point-estimate performance (median, q50) on test ----
    point_pred = q_test["q50"].values
    rmse = np.sqrt(mean_squared_error(y_test, point_pred))
    mae = mean_absolute_error(y_test, point_pred)
    mape = mean_absolute_percentage_error(y_test, point_pred)
    r2 = r2_score(y_test, point_pred)
    print("\n=== Point estimate (median, q50) on held-out TEST split ===")
    print(f"RMSE={rmse:.0f}  MAE={mae:.0f}  MAPE={mape:.4f}  R2={r2:.4f}")

    # ---- interval calibration: raw vs conformal-corrected ----
    print("\n=== Interval calibration (held-out TEST split) ===")
    rows = []
    for lo_q, hi_q, nominal in INTERVALS:
        lo_col, hi_col = f"q{int(lo_q*100):02d}", f"q{int(hi_q*100):02d}"

        # raw (uncalibrated) quantile-regression interval
        raw_cov, raw_width = coverage_and_width(y_test.values, q_test[lo_col].values, q_test[hi_col].values)

        # CQR: margin fit on calibration split, applied to test split
        margin = cqr_margin_for_level(y_cal.values, q_cal[lo_col].values, q_cal[hi_col].values, nominal)
        lo_adj = np.maximum(q_test[lo_col].values - margin, 0)
        hi_adj = q_test[hi_col].values + margin
        cqr_cov, cqr_width = coverage_and_width(y_test.values, lo_adj, hi_adj)

        rows.append({
            "interval": f"[{lo_col},{hi_col}]", "nominal_coverage": nominal,
            "raw_coverage": raw_cov, "raw_avg_width": raw_width,
            "cqr_margin": margin, "cqr_coverage": cqr_cov, "cqr_avg_width": cqr_width,
        })
        print(f"nominal={nominal:.0%}  raw: cov={raw_cov:.3f} width=${raw_width:,.0f}   "
              f"CQR: margin=${margin:,.0f} cov={cqr_cov:.3f} width=${cqr_width:,.0f}")

    calib_df = pd.DataFrame(rows)
    calib_df.to_csv("interval_calibration_results.csv", index=False)

    # ---- coverage by price tercile, for the 80% interval, CQR-adjusted ----
    print("\n=== 80% CQR interval coverage by price tercile (TEST split) ===")
    lo_col, hi_col = "q10", "q90"
    margin80 = cqr_margin_for_level(y_cal.values, q_cal[lo_col].values, q_cal[hi_col].values, 0.80)
    lo_adj = np.maximum(q_test[lo_col].values - margin80, 0)
    hi_adj = q_test[hi_col].values + margin80
    tercile = pd.qcut(y_test.values, 3, labels=["low", "mid", "high"])
    tercile_df = pd.DataFrame({
        "tercile": tercile, "y": y_test.values, "lo": lo_adj, "hi": hi_adj,
    })
    tercile_df["covered"] = (tercile_df.y >= tercile_df.lo) & (tercile_df.y <= tercile_df.hi)
    tercile_df["width"] = tercile_df.hi - tercile_df.lo
    tercile_summary = tercile_df.groupby("tercile", observed=True).agg(
        n=("y", "size"), price_range=("y", lambda s: f"${s.min():,.0f}-${s.max():,.0f}"),
        coverage=("covered", "mean"), avg_width=("width", "mean"),
    )
    print(tercile_summary.to_string())
    tercile_summary.to_csv("coverage_by_price_tercile.csv")

    # ---- save full test-set predictions for inspection ----
    out = X_test.copy()
    out["property_id"] = groups.iloc[test_idx].values
    out["actual_price"] = y_test.values
    for c in q_test.columns:
        out[c] = q_test[c].values
    out["interval80_lo_cqr"] = lo_adj
    out["interval80_hi_cqr"] = hi_adj
    out.to_csv("test_predictions_with_intervals.csv", index=False)
    print("\nSaved interval_calibration_results.csv, coverage_by_price_tercile.csv, test_predictions_with_intervals.csv")
