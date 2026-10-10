# EDA notes — Stage 3 (train rows only, n=3900)

## Feature shape
- `tenure_months`: near-uniform 1-72, no skew (skew=0.01).
- `monthly_charges`: mildly right-skewed, range 15-164.
- `total_charges`: right-skewed (skew 0.53), correlated with tenure x monthly_charges but with
  residual noise (credits/discounts/prorations) — kept as its own feature, not recomputed.
- `avg_daily_usage_mb`: heavy right tail (skew 20+ on raw scale), 93 points > 3IQR, top 0.1%
  above 2000 MB/day vs a median of 12.4 — candidate for log1p transform for the linear proxy
  at Stage 5; not impossible so not nulled.
- `age`: after Stage 1 cleaning (impossible 0/250 -> NaN), clean 18-85 range, mild spread.
- `support_calls_90d`: count 0-7, right-skewed, zero/one-inflated (2557/3900 rows have 0-1 calls).

## Feature x target (train only, decile/level tables in code/03_stage3_leak_eda.py output)
- `tenure_months`: monotonic-ish decrease in churn with tenure, 13.9% (0-8mo) -> 4.7% (66-72mo).
  Classic early-tenure churn risk.
- `monthly_charges`: monotonic-ish increase, 7.2% (cheapest decile) -> 13.3% (priciest decile).
- `support_calls_90d`: strongly monotonic, 7.0% (0-1 calls) -> 10.1% (2) -> 13.0% (3) -> 21.6%
  (4-7). The cleanest, strongest univariate driver with no timing ambiguity.
- `contract_type`: month-to-month 12.3% vs one-year 3.9% / two-year 4.5% — large, matches
  well-known telco churn behaviour (no commitment = easy to leave). Strong, legit driver.
- `payment_method`: electronic check 11.3% vs mailed check/credit card/bank transfer ~7.8-8.0%
  — moderate effect, consistent with the classic Telco-churn dataset pattern this file resembles.
- `avg_daily_usage_mb`: flat by decile (all bins 7.7%-10.0%, no trend) despite being the most
  extreme-valued raw column — usage level alone barely separates churners; may still help a tree
  in interaction with other features, but is not a strong marginal driver.
- `age`, `region`: weak, noisy spreads (age no clear trend; region 5.9%-12.0% across 20 levels
  with ~200 rows each — likely partly sampling noise, watch as a Stage 9 slice rather than trust
  as a driver).

## Redundancy
- `last_invoice_amount` vs `monthly_charges`: Spearman 0.97 (**above the 0.9 redundancy cut**).
  Combined with `last_invoice_amount`'s unresolved leak-timing risk (see `leak_table.csv`) and
  its 18% missingness, the simpler and cleaner `monthly_charges` is preferred; `last_invoice_amount`
  is carried to Stage 4/6 for an ablation rather than assumed necessary.
- `total_charges` vs `tenure_months`: Spearman 0.85 (below the 0.9 cut, kept as separate features;
  a GBDT/linear model can use the residual signal — total_charges is not exactly tenure x
  monthly_charges, it carries extra information such as historical plan changes).
- No other pair exceeds |Spearman| 0.9.

## Stability
- Churn rate by signup-month (Stage 2 check): 4.7%-12.8%, no monotonic trend — no evidence of
  drift over the observed window. Contract-type and support-call effects were spot-checked on
  two halves of train (first vs second half by row order) and point the same direction in both
  halves (no sign flip found).

## Slices to watch at Stage 9
- Region (20 levels, ~200 rows/level in train — individual regions will have wide CIs; report
  the 2-3 furthest-from-average regions but do not over-interpret single-region swings).
- Very short tenure (<=8 months) x month-to-month contract: expected highest-risk segment,
  worth checking the model actually ranks it correctly.
- Rows with missing `last_invoice_amount` (18% of train, differential by churn) — check the
  model isn't relying on the missingness indicator as a proxy for the dropped leak columns.

## Label-quality check
Skipped (optional per `ds-data-audit`): time budget prioritized toward the joint search; target
is a clean binary flag with no evidence of noise (no duplicate customer_id with conflicting
labels found in Stage 1 audit — 0 conflicting rows among 300 rows sharing a key).

## Feature ideas for Stage 6
- `high_risk_contract_x_tenure`: interaction of month-to-month contract and tenure <= 8 months.
- `support_calls_per_tenure_month`: support_calls_90d normalized loosely by tenure (careful:
  90d window vs lifetime tenure is a unit mismatch, treat as a ratio feature, not a rate law).
- `total_charges_residual`: total_charges - tenure_months*monthly_charges (captures
  discounts/credits/plan-change history beyond the two raw fields).
- `missing_last_invoice_flag`: binary indicator, tested in ablation alongside/instead of the
  raw `last_invoice_amount` value.
