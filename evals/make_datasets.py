"""Generate synthetic evaluation datasets with planted defects.

Each dataset is written to evals/datasets/ and its hidden answer key to
evals/answer_keys/. The answer keys are used only for grading test runs; the
agent under test never sees them.

Run:  python evals/make_datasets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
KEYS = ROOT / "answer_keys"
DATA.mkdir(exist_ok=True)
KEYS.mkdir(exist_ok=True)

rng = np.random.default_rng(20260927)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC without sklearn, so the generator has no ML dependency."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# 1. Churn: binary classification with leakage, sentinels, dirt, imbalance
# --------------------------------------------------------------------------
def make_churn(n: int = 6000) -> None:
    tenure = rng.integers(1, 73, n)
    monthly = np.clip(rng.normal(70, 25, n), 15, 200).round(2)
    total = (tenure * monthly * rng.normal(1, 0.05, n)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    support_calls = rng.poisson(1.2, n)
    usage = np.exp(rng.normal(2.5, 0.6, n)).round(2)
    region = rng.choice([f"R{i:02d}" for i in range(20)], n)
    age = rng.integers(18, 86, n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")

    logit = (
        -2.9
        + 1.2 * (contract == "Month-to-month")
        + 0.5 * (payment == "Electronic check")
        + 0.35 * support_calls
        - 0.02 * tenure
        + 0.008 * (monthly - 70)
        - 0.15 * np.log1p(usage)
        + rng.normal(0, 0.3, n)
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p).astype(int)
    honest_auc = _auc(p, churn)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract_type": contract,
            "payment_method": payment,
            "support_calls_90d": support_calls,
            "avg_daily_usage_mb": usage,
            "region": region,
            "age": age,
            "churned": churn,
        }
    )

    # Planted defect: informative missingness in last_invoice_amount. Missingness depends on the
    # pre-outcome churn risk p (billing problems cluster among at-risk customers), not on the realised
    # label, so the indicator is a legitimate feature rather than a leak.
    invoice = (monthly * rng.normal(1, 0.08, n)).round(2)
    miss_p = np.clip(0.02 + 2.0 * p, 0, 0.9)
    invoice = np.where(rng.random(n) < miss_p, np.nan, invoice)
    df["last_invoice_amount"] = invoice

    # Planted defect: sentinel -999 and true NaN in total_charges.
    idx = rng.choice(n, int(0.06 * n), replace=False)
    df.loc[idx, "total_charges"] = -999
    idx2 = rng.choice(np.setdiff1d(np.arange(n), idx), int(0.04 * n), replace=False)
    df.loc[idx2, "total_charges"] = np.nan

    # Planted defect: inconsistent category spellings.
    idx = rng.choice(n, int(0.15 * n), replace=False)
    variants = {
        "Month-to-month": ["month-to-month", "MONTH-TO-MONTH", "Month-to-month "],
        "One year": ["one year", "One Year", " One year"],
        "Two year": ["two year", "TWO YEAR", "Two year "],
    }
    df.loc[idx, "contract_type"] = [rng.choice(variants[v]) for v in df.loc[idx, "contract_type"]]

    # Planted defect: impossible ages.
    idx = rng.choice(n, int(0.01 * n), replace=False)
    df.loc[idx, "age"] = rng.choice([0, 250], len(idx))

    # Planted defect: extreme outliers in usage (data-entry x100).
    idx = rng.choice(n, int(0.005 * n), replace=False)
    df.loc[idx, "avg_daily_usage_mb"] = df.loc[idx, "avg_daily_usage_mb"] * 100

    # Planted leak 1: churn_reason is only populated after a customer churns.
    reasons = ["price", "competitor", "service", "moved", "other"]
    df["churn_reason"] = np.where(churn == 1, rng.choice(reasons, n), "")

    # Planted leak 2: account_status_code encodes the outcome (with a little noise).
    status = np.where(churn == 1, "C", rng.choice(["A", "S"], n, p=[0.9, 0.1]))
    flip = rng.random(n) < 0.02
    status = np.where(flip, rng.choice(["A", "C", "S"], n), status)
    df["account_status_code"] = status

    # Planted defect: exact duplicate rows.
    dup_idx = rng.choice(n, 150, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    df.to_csv(DATA / "churn.csv", index=False)
    key = {
        "task": "binary classification, target=churned",
        "positive_rate": round(float(churn.mean()), 4),
        "honest_ceiling_roc_auc": round(honest_auc, 3),
        "note_on_ceiling": "AUC of the true generating probability vs the label. Any reported AUC materially above this means leakage.",
        "leak_columns": ["churn_reason", "account_status_code"],
        "sentinel_values": {"total_charges": -999},
        "true_missing": {
            "total_charges": "4% MCAR",
            "last_invoice_amount": "informative: missing rate rises with pre-outcome churn risk (0.02 + 2.0*p, capped at 0.9); NOT a leak, the missing indicator is a legitimate feature",
        },
        "category_inconsistency": ["contract_type"],
        "impossible_values": {"age": "0 or 250 in ~1% of rows"},
        "outliers": {"avg_daily_usage_mb": "~0.5% multiplied by 100"},
        "exact_duplicate_rows": 150,
        "id_column": "customer_id",
        "true_drivers": ["contract_type", "payment_method", "support_calls_90d", "tenure_months", "monthly_charges", "avg_daily_usage_mb"],
        "business_constraint": "retention team can call ~10% of customers; precision/recall at top 10% matters, not accuracy",
    }
    (KEYS / "churn.json").write_text(json.dumps(key, indent=2))


# --------------------------------------------------------------------------
# 2. House prices: regression with skewed target, leak, group leakage, drift
# --------------------------------------------------------------------------
def make_house_prices(n: int = 4000) -> None:
    sqft = np.exp(rng.normal(np.log(1800), 0.35, n)).round(0)
    bedrooms = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + rng.normal(0, 0.5, n)), 1, 4).astype(int)
    year_built = rng.integers(1900, 2024, n)
    lot = np.exp(rng.normal(np.log(6000), 0.5, n)).round(0)
    neigh_p = 1 / np.arange(1, 31) ** 0.9
    neigh_p /= neigh_p.sum()
    neighborhood = rng.choice([f"N{i:02d}" for i in range(30)], n, p=neigh_p)
    neigh_effect = dict(zip([f"N{i:02d}" for i in range(30)], rng.normal(0, 0.25, 30)))
    distance = np.exp(rng.normal(np.log(8), 0.6, n)).round(1)
    cond_levels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    cond_idx = rng.choice(5, n, p=[0.05, 0.15, 0.45, 0.25, 0.10])
    condition = np.array(cond_levels)[cond_idx]
    garage = rng.choice(["Yes", "No"], n, p=[0.7, 0.3])
    sold = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1460, n), unit="D")
    years_since = (sold - pd.Timestamp("2021-01-01")).days / 365.25

    log_price = (
        11.2
        + 0.75 * np.log(sqft / 1800)
        + 0.04 * bedrooms
        + 0.06 * bathrooms
        + 0.002 * (year_built - 1980)
        + 0.08 * np.log(lot / 6000)
        + np.array([neigh_effect[v] for v in neighborhood])
        - 0.12 * np.log(distance / 8)
        + 0.07 * (cond_idx - 2)
        + 0.05 * (garage == "Yes")
        + 0.045 * years_since  # planted drift: prices rise ~4.5%/year
    )
    sale_noise = rng.normal(0, 0.12, n)
    expected_price = np.exp(log_price)  # what an assessor could know before the sale
    price = np.exp(log_price + sale_noise).round(-2)

    df = pd.DataFrame(
        {
            "listing_id": [f"L{i:06d}" for i in range(n)],
            "property_id": [f"P{i:06d}" for i in range(n)],
            "sold_date": sold.strftime("%Y-%m-%d"),
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size_sqft": lot,
            "neighborhood": neighborhood,
            "distance_to_center_km": distance,
            "condition": condition,
            "garage": garage,
            # Assessed before the sale from the property's expected value, with independent error.
            "tax_assessed_value": (expected_price * 0.85 * rng.normal(1, 0.10, n)).round(-2),
            "price": price,
        }
    )

    # Planted leak: price_per_sqft is derived from the target.
    df["price_per_sqft"] = (df["price"] / df["sqft"]).round(2)

    # Planted defect: MCAR missing lot size.
    idx = rng.choice(n, int(0.20 * n), replace=False)
    df.loc[idx, "lot_size_sqft"] = np.nan

    # Planted defect: inconsistent garage coding incl. blanks.
    idx = rng.choice(n, int(0.12 * n), replace=False)
    df.loc[idx, "garage"] = [
        {"Yes": rng.choice(["Y", "yes", "YES"]), "No": rng.choice(["N", "no", ""])}[v] for v in df.loc[idx, "garage"]
    ]

    # Planted defect: impossible year_built and sqft outliers.
    idx = rng.choice(n, int(0.005 * n), replace=False)
    df.loc[idx, "year_built"] = rng.choice([1800, 2090], len(idx))
    idx = rng.choice(n, int(0.005 * n), replace=False)
    df.loc[idx, "sqft"] = rng.choice([50, 100000], len(idx))

    # Planted group leakage: 8% of properties sold twice (repeat sales).
    # Only properties first sold before 2023 are resold, so every resale (+200..700 days) stays in the data window.
    rep_idx = rng.choice(np.flatnonzero(sold < pd.Timestamp("2023-01-01")), int(0.08 * n), replace=False)
    repeats = df.iloc[rep_idx].copy()
    repeats["listing_id"] = [f"L{n + i:06d}" for i in range(len(repeats))]
    later = pd.to_datetime(repeats["sold_date"]) + pd.to_timedelta(rng.integers(200, 700, len(repeats)), unit="D")
    repeats["sold_date"] = later.dt.strftime("%Y-%m-%d")
    repeats["price"] = (repeats["price"] * rng.normal(1.06, 0.04, len(repeats))).round(-2)
    repeats["price_per_sqft"] = (repeats["price"] / repeats["sqft"]).round(2)
    df = pd.concat([df, repeats], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)

    df.to_csv(DATA / "house_prices.csv", index=False)
    key = {
        "task": "regression, target=price",
        "target_distribution": "log-normal (right-skewed); log-transform or a robust loss is appropriate",
        "honest_noise_sd_log": 0.12,
        "note_on_ceiling": "Irreducible noise is ~12% (log scale). MAPE far below ~9% or R2 near 1.0 means leakage.",
        "leak_columns": ["price_per_sqft"],
        "strong_legit_feature": "tax_assessed_value (assessed before the sale from expected value with independent 10% error; keep). A good run asks when it was recorded relative to the sale.",
        "group_key": "property_id (8% of properties appear twice; must not straddle train/test)",
        "time_drift": "prices rise ~4.5% per year via sold_date; a time-aware split or a time feature is needed",
        "true_missing": {"lot_size_sqft": "20% MCAR"},
        "category_inconsistency": ["garage (Yes/Y/yes/YES, No/N/no/blank)"],
        "ordinal_feature": "condition (Poor < Fair < Good < Very Good < Excellent)",
        "rare_categories": "neighborhood has 30 levels with a Zipf-like tail",
        "impossible_values": {"year_built": "1800 or 2090", "sqft": "50 or 100000"},
        "business_constraint": "pricing tool needs MAE and prediction intervals for new listings",
    }
    (KEYS / "house_prices.json").write_text(json.dumps(key, indent=2))


# --------------------------------------------------------------------------
# 3. Store sales: weekly multi-series forecasting with gaps, shift, leak
# --------------------------------------------------------------------------
def make_store_sales(n_stores: int = 20, n_weeks: int = 104) -> None:
    weeks = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    rows = []
    base = np.exp(rng.normal(np.log(20000), 0.4, n_stores))
    for s in range(n_stores):
        store = f"S{s:02d}"
        trend = rng.normal(0.002, 0.001)
        seas_amp = rng.uniform(0.1, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        promo = rng.random(n_weeks) < 0.15
        temp = 15 + 12 * np.sin(2 * np.pi * (np.arange(n_weeks) / 52) - np.pi / 2) + rng.normal(0, 3, n_weeks)
        holiday = np.isin(weeks.isocalendar().week.values, [1, 14, 27, 47, 51, 52])
        for t, wk in enumerate(weeks):
            level = base[s] * (1 + trend) ** t
            seas = 1 + seas_amp * np.sin(2 * np.pi * t / 52 + phase)
            eff = 1 + 0.25 * promo[t] + 0.15 * holiday[t]
            if store == "S07" and t >= 60:
                eff *= 1.5  # planted level shift
            sales = level * seas * eff * rng.normal(1, 0.06)
            if store == "S19":
                sales = sales * 0.05 * (rng.random() < 0.35)  # planted intermittent demand
            sales = round(max(sales, 0), 0)
            rows.append(
                {
                    "store_id": store,
                    "week_start": wk.strftime("%Y-%m-%d"),
                    "promo": int(promo[t]),
                    "holiday_week": int(holiday[t]),
                    "avg_temp_c": round(float(temp[t]), 1),
                    "sales": sales,
                    "inventory_end": round(max(level * 2.2 - sales + rng.normal(0, 500), 0), 0),
                }
            )
    df = pd.DataFrame(rows)

    # Planted defect: missing weeks for S03.
    drop = df[(df["store_id"] == "S03") & (df["week_start"].isin(["2023-06-05", "2023-06-12", "2024-02-19"]))].index
    df = df.drop(drop)

    # Planted defect: sentinel -1 sales.
    idx = rng.choice(df.index, 5, replace=False)
    df.loc[idx, "sales"] = -1

    # Planted defect: duplicated (store, week) rows.
    dup = df.sample(4, random_state=3)
    df = pd.concat([df, dup], ignore_index=True).sort_values(["store_id", "week_start"]).reset_index(drop=True)

    df.to_csv(DATA / "store_sales.csv", index=False)
    key = {
        "task": "multi-series weekly forecasting, horizon=8 weeks, target=sales per store",
        "n_series": n_stores,
        "n_weeks": n_weeks,
        "leak_columns": [
            "inventory_end for the week being forecast (it is a function of that week's sales). Using inventory_end at or before the forecast origin is legitimate."
        ],
        "future_known_covariates": ["promo", "holiday_week"],
        "unknown_future_covariate": "avg_temp_c (would need its own forecast or a climatology)",
        "gaps": "S03 is missing weeks 2023-06-05, 2023-06-12, 2024-02-19",
        "sentinel_values": {"sales": -1, "count": 5},
        "duplicate_keys": 4,
        "level_shift": "S07 x1.5 from week index 60 (2024-02-26) onward",
        "intermittent_series": "S19 (mostly zeros)",
        "seasonality": "yearly (52-week), trend, promo +25%, holiday +15%",
        "required_validation": "rolling-origin / time-based backtest with 8-week horizon; random K-fold is wrong",
        "sensible_baselines": "seasonal naive (lag 52), last value, moving average",
        "business_constraint": "inventory planning; per-store 8-week forecast with an accuracy statement",
    }
    (KEYS / "store_sales.json").write_text(json.dumps(key, indent=2))


if __name__ == "__main__":
    make_churn()
    make_house_prices()
    make_store_sales()
    for f in sorted(DATA.glob("*.csv")):
        d = pd.read_csv(f)
        print(f"{f.name}: {d.shape[0]} rows x {d.shape[1]} cols")
    print("answer keys:", [k.name for k in sorted(KEYS.glob('*.json'))])
