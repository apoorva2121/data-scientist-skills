# Problem statement: House pricing tool

Date: 2026-09-27   Author: data-scientist agent (apoorva.dixit21@gmail.com)   Status of assumptions: 6 ASSUMED lines open

| Field | Value | Source |
|---|---|---|
| Unit of analysis | One completed sale (a listing that sold); `listing_id` is the row key, `property_id` repeats when the same physical property sold more than once (320 of 4320 rows are repeat properties) | data |
| Target | `price` (sale price, USD). Modeled as `log(price)` internally for variance stabilization (raw skew 1.59, log skew 0.10), back-transformed for reporting. | data + ASSUMED (log-scale modeling) |
| Prediction moment | At new-listing time, before sale: only attributes knowable from the property record and listing (sqft, beds, baths, year_built, lot size, neighborhood, distance to center, condition, garage, tax-assessed value) are usable. `price_per_sqft` is derived from the realized sale price and is NOT available at listing time. | ASSUMED, matches stated use case "pricing tool for new listings" |
| Decision fed | An agent/tool suggesting a list price (or price range) for a new listing; the point estimate anchors the suggestion and the interval bounds how confidently to quote it. | ASSUMED from task description |
| Problem type | Regression (supervised) | data |
| Primary metric + reason | MAE on price, in dollars, because the business action is "set a list price" and dollar error is directly interpretable to a listing agent; also report RMSE (penalizes large misses, relevant since tail listings are expensive) and MAPE (comparability across price levels). | ASSUMED |
| Secondary metrics | Prediction-interval coverage (target 90%) and mean interval width, from split-conformal residuals; per-neighborhood and per-price-decile error slices. | this skill (interval requested by user) |
| Baseline to beat | Neighborhood-median $/sqft rule: predicted_price = median(price/sqft \| neighborhood) x sqft. Represents what a naive comp-based agent does today. | ASSUMED (no incumbent model/process given) |
| Success threshold | Beat the baseline's MAE by more than the CV noise floor, and achieve interval coverage within [87%, 93%] of the nominal 90%. | ASSUMED |
| Constraints | No hard latency constraint (batch/interactive scoring of single rows is fine); moderate interpretability desirable (agents will ask "why this price") but not regulatory; no GPU; retraining cadence ASSUMED quarterly as new sales accrue. | ASSUMED |
| Data size | 4320 rows, 15 columns, 4000 unique properties (320 repeat-sale properties), single-family-style records, sales dated 2021-01-01 to 2024-12-30 (4 years) | data |
| Time and compute | Hard budget: 25 minutes total wall-clock for this task, CPU only, no GPU, no package installs (MAPIE unavailable -> conformal implemented by hand). Budget split per skill: ~6 min stages 0-3, ~9 min stages 4-7, ~6 min stage 8, ~4 min stages 9-10. | user instruction |

Needs ML at all? Yes with caveats: the naive $/sqft-by-neighborhood rule is a real, cheap baseline and is
implemented and measured first; if no model beats it beyond the noise floor the recommendation
reverts to that rule (see Stage 7 stop signal). Given 8 numeric/categorical drivers with
plausible non-linear and interaction effects (condition x age, neighborhood x sqft), a fitted
model is expected to beat a single ratio rule.

## Open ASSUMED lines (would confirm with data/business owner if reachable)
1. Log-scale modeling of price is appropriate (no stated floor/cap on price).
2. Point estimate + 90% interval is the deliverable shape (not a full distribution or quantiles).
3. No incumbent baseline model exists; the neighborhood $/sqft rule is a fair stand-in.
4. Quarterly retraining cadence; no stated latency SLA.
5. `price_per_sqft` is confirmed a post-sale-derived column and must be excluded as a feature.
6. Repeat sales of the same `property_id` are independent listings across time (not renovations that would change the unit of analysis) — treated as separate rows but grouped in the split.
