# Problem statement: Customer churn risk scoring for retention calling
Date: 2026-09-27   Author: data-scientist skill (automated)   Status of assumptions: 6 ASSUMED lines open (user unavailable; time-boxed 25 min run)

| Field | Value | Source |
|---|---|---|
| Unit of analysis | One row = one customer, snapshot as of the data pull | ASSUMED |
| Target | `churned` (binary 0/1). Window/observation period not documented in the file. | data; window ASSUMED |
| Prediction moment | ASSUMED: the monthly snapshot date, before any post-cancellation fields (`churn_reason`, cancelled `account_status_code`) are written | ASSUMED |
| Decision fed | Retention team calls customers predicted highest-risk; hard capacity = 10% of the customer base per month | user |
| Problem type | Binary classification -> supervised path | user |
| Primary metric | Precision@10% (= capture rate among the top 10% by predicted risk), because the action is capacity-limited to 10% of customers, so ranking quality within that budget is what matters, not overall accuracy | derived from decision fed |
| Secondary metrics | Average precision (PR-AUC) — target is imbalanced (~8.9% positive after cleaning), ROC-AUC is only weakly informative; recall captured at 10%; lift over base rate; calibration (Brier, reliability) in case scores are used as probabilities later | ASSUMED |
| Baseline to beat | Trivial prior-rate baseline; a 2-line business rule (month-to-month contract + high support calls) scored the same way | ASSUMED (no existing process given) |
| Success threshold | ASSUMED: precision@10% at least 3x the base rate (~26%+), i.e. a materially better call list than calling at random | ASSUMED |
| Constraints | ASSUMED: batch monthly scoring (no hard latency limit); retention team wants driver explanations, so at least one interpretable output is required; CPU only, no GPU, no new package installs | ASSUMED |
| Data size | 6,150 raw rows / 15 columns; 150 exact duplicate rows (see cleaning log); ~532 positive raw rows (8.65%) | data |
| Time and compute | Hard 25-minute wall-clock budget for the whole project -> compact mode; CPU only; scikit-learn 1.7.2, lightgbm 4.7, xgboost 3.4, catboost, optuna 5.0, imbalanced-learn, statsmodels available, no installs permitted | user |

Needs ML at all? Likely yes but not certain: several individually weak/moderate signals (contract type,
support calls, tenure, usage) suggest a single 2-line rule may not rank as well as a learned combination.
This is tested directly at Stage 7 by comparing a business-rule baseline against screened model families
under the same primary metric, rather than assumed.
