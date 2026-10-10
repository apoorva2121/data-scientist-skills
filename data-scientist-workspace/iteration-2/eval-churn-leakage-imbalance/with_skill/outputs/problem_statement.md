# Problem statement: Churn propensity for retention calling

Date: 2026-09-27   Author: data-scientist (automated, user unavailable)   Status of assumptions: 5 ASSUMED lines open

| Field | Value | Source |
|---|---|---|
| Unit of analysis | One row = one customer account (`customer_id`) at the current scoring snapshot | data |
| Target | `churned` (0/1). Definition/window not documented by source. ASSUMED: "churned by the time this snapshot was pulled" | ASSUMED |
| Prediction moment | ASSUMED: the monthly scoring run, using only information known about the customer as of that run (i.e. not information recorded at/after cancellation) | ASSUMED |
| Decision fed | Retention team calls customers predicted likely to churn; capacity = 10% of the customer base per month | user |
| Problem type | binary classification -> supervised path | user |
| Primary metric | Precision@10% (precision among the top decile ranked by predicted churn probability), because the action is capacity-constrained to ~10% of customers regardless of how many are actually at risk | derived from decision fed |
| Secondary metrics | PR-AUC (imbalance-appropriate discrimination), ROC-AUC, recall captured@10%, calibration (Brier/reliability), per-region and per-contract-type slices | this skill |
| Baseline to beat | (a) trivial: call a random 10% (expected precision = base rate 8.65%); (b) simple rule: rank by `support_calls_90d` desc, ties by `contract_type`=month-to-month | ASSUMED - no documented current process |
| Success threshold | ASSUMED: precision@10% >= 3x base rate (~26%), i.e. the called list is at least 3x as likely to churn as a random customer | ASSUMED |
| Constraints | ASSUMED: batch/monthly scoring (no low-latency requirement); retention team wants interpretable drivers ("what actually drives churn") -> favor a model family that supports reliable importances; no fairness/regulatory constraint stated | ASSUMED |
| Data size | 6,150 rows raw / 6,000 after removing 150 exact duplicate rows; 15 columns; 532 positives (8.65% churn rate) before dedup, 508 after | data |
| Time and compute | 25 minutes total wall-clock, CPU only, no GPU. This forces a small search (few candidates, 3x3 CV, no Optuna study) | user |

Needs ML at all? Yes. Churn correlates with several weak-to-moderate signals (contract type, tenure,
support calls, charges) with no single rule dominating (checked in Stage 3); a ranking model should
beat a single-column heuristic enough to matter for a capacity-constrained call list, and the
"what drives churn" ask requires a fitted model's importances, not just a lookup.

## Open ASSUMED lines to confirm with the data owner
1. Target definition/observation window for `churned` (exact churn event and cutoff date).
2. Whether `last_invoice_amount` is billed at/around the prediction moment (kept as a feature) or is
   the final closing invoice generated at cancellation (would be a leak) - see Stage 3 leak table.
3. Success threshold (3x base rate) is a placeholder business bar, not a stated target.
4. No documented current retention process/baseline precision existed to compare against.
5. Whether the 150 exact-duplicate customer_id rows are double-counted billing extracts (dropped as
   duplicates here) or two genuinely distinct events - treated as duplicates since all columns match exactly.
