# Stage 3 EDA notes (train rows only, n=2951)

## Shape
- price: right-skewed (skew 1.59 on full data); log(price) skew ~0.10 -> model on log scale.
- sqft, lot_size_sqft, tax_assessed_value: right-skewed, long tails toward large/expensive
  properties; log-transform helps the linear proxy, irrelevant to tree models.
- neighborhood: 30 levels, top 3 (N00,N01,N02) cover ~40% of rows; several levels <50 rows
  (rare-level handling / target-encoding smoothing needed at Stage 6).

## Feature vs target (Spearman |r| on train)
tax_assessed_value 0.94, sqft 0.72, bedrooms 0.64, bathrooms 0.55, neighborhood (eta) 0.48,
year_built 0.15, distance_to_center_km 0.13, condition 0.11, lot_size_sqft 0.07, garage 0.04.
tax_assessed_value and sqft dominate, as expected for a hedonic price model; bedrooms/bathrooms
correlate with sqft (redundancy check below).

## Redundancy
- bedrooms vs sqft: Spearman 0.55; bathrooms vs sqft: 0.58 - moderate, not >0.9, keep both (a
  4-bed/1600sqft house prices differently than a 2-bed/1600sqft house).
- tax_assessed_value vs sqft: 0.71 - moderate; tax_assessed_value carries information beyond sqft
  (condition, lot, location already baked in by the assessor) so both are kept; VIF check deferred
  to Stage 6/7 for the linear family only.

## Stability / drift
- Adversarial validation, train (2021-2023) vs test (2024), HistGradientBoostingClassifier,
  5-fold CV AUC = 0.608. Expected and acceptable for a time-based split (measures market drift,
  not a leak): no single feature separates train/test above AUC 0.53 (checked individually,
  max = tax_assessed_value 0.525), so the drift is diffuse (broad price/mix shift over time),
  not one leaking column. Confirms the time-based split is doing its job; recommend the final
  pipeline include sold_year/sold_month as a feature so the model can pick up the trend it will
  be asked to extrapolate at Stage 9/10.

## Slices to watch at Stage 9
- Neighborhood (30 levels, uneven counts) - per-neighborhood error.
- Price decile - check under/over-prediction at the top decile (expensive homes, thin data).
- Condition (Poor/Fair/Good/Very Good/Excellent) - fewer Poor/Excellent rows.
- Sold-date quarter within test (May-Dec 2024) - check for within-test drift.

## Feature ideas for Stage 6
house_age = sold_year - year_built; total_rooms = bedrooms + bathrooms; log1p(sqft),
log1p(lot_size_sqft), log1p(tax_assessed_value); sold_year, sold_month (trend/seasonality);
neighborhood target-encoding (in-fold) as an alternative to one-hot for the linear proxy.
