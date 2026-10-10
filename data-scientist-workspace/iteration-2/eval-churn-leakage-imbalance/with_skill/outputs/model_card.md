# Model card: churn propensity for retention calling

## Intended use
Rank customers monthly by predicted churn probability so the retention team can call the top 10%
(their fixed monthly capacity). Not intended for automated action (no calls are placed by the
model) and not validated for any use beyond ranking this customer base for this call program.

## Data
6,150 raw customer rows (6,000 after removing 150 exact-duplicate rows), 1 snapshot, 8.65-8.67%
historical churn rate. Features: tenure, monthly/total charges, contract type, payment method,
support calls (90d), average daily usage, region, age, last invoice amount, plus two engineered
features (charge_gap, is_new_customer). Two columns were dropped as post-outcome leaks:
`churn_reason` and `account_status_code` (see decision_log.md Stage 3).

## Model
Regularised logistic regression (C=0.01, class_weight="balanced"), one-hot encoded categoricals,
median imputation with missingness indicators, standardized numerics. Chosen over LightGBM/XGBoost,
which underperformed by a wide, consistent margin at every screening stage (see decision_log.md
Stages 7-8) - a "small/mostly-linear signal" pattern.

## Metrics (test set, n=1200, 104 positives, 95% bootstrap CI)
- Precision@10% (primary): 0.267 [0.192, 0.350] - i.e. of the 120 customers called, ~32 are expected
  to actually churn, a 3.08x lift over the 8.67% base rate.
- Recall captured@10%: 0.308 (about 31% of all churners this period are in the top decile).
- ROC-AUC: 0.718 [0.667, 0.768]. PR-AUC: 0.213 [0.161, 0.301]. Brier score: 0.223 (uncalibrated -
  probabilities are class-weight-distorted; only the ranking is validated, not the raw probability).

## Top drivers (logistic coefficients + permutation importance agree)
1. Contract type - month-to-month raises risk, one/two-year contracts lower it (largest effect).
2. Tenure - longer tenure lowers risk.
3. Support calls in the last 90 days - more calls raises risk.
Secondary: payment method (electronic check higher risk), last invoice amount, region (weak, a few
regions stand out but not strongly).

## Limitations
- Weakest for One-year-contract customers (ROC-AUC 0.53 on that slice, near chance) - the model
  gives this group almost no useful ranking signal.
- False negatives are concentrated among long-tenure, low-support-call customers who churn without
  any behavioral warning in the available features - a feature gap (see report.md next steps).
- `last_invoice_amount` timing is an unconfirmed ASSUMED input (see problem_statement.md); if it
  turns out to be a post-cancellation final invoice, it must be removed and the model rebuilt.
- Probabilities are not calibrated (class-weighted); only the rank ordering / top-10% cut is
  validated. Do not use the raw score as a churn probability in other downstream uses.
- Built under a 25-minute time budget: no nested hyperparameter search (Optuna), no KNN/iterative
  imputation screen, no interaction features, no foundation-model screen (TabPFN licence and hosted-
  inference data-location were not clearable with the user unavailable). See report.md next steps.

## Ethical / fairness notes
No protected attributes (race, gender, etc.) are present in the data, so no fairness metric was
computed. `region` and `age` are proxies worth a fairness review before any pricing or eligibility
use (this model is for call prioritization only, not pricing or eligibility).

## Monitoring plan
- **Input drift**: monthly PSI/KS on tenure_months, monthly_charges, support_calls_90d,
  avg_daily_usage_mb, and the contract_type distribution, vs the training snapshot.
- **Prediction drift**: monthly distribution of predicted probabilities and the score at the 10%
  cutoff; alert if the cutoff score moves more than ~2x its historical month-to-month movement.
- **Delayed-label performance**: once next month's actual churn outcomes are known, recompute
  precision@10% and ROC-AUC on that month's calls and compare against the test-set CI above; a
  precision@10% below the CI lower bound (0.19) for 2 consecutive months triggers a review.
- **Data validation**: re-run the Stage 1 cleaning rules (sentinel checks for total_charges==-999,
  age<=0 or >=120, contract_type spelling) on every incoming batch; a spike in any of these should
  block scoring, not silently pass through.
- **Retraining trigger**: retrain when 6 months have passed, or delayed-label precision@10% breaches
  the trigger above, or the churn base rate shifts by more than 2 points from 8.67%.
