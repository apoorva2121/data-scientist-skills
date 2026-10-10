# Customer churn: model report
Status: Stages 0-10 all passed (25-minute compact-mode run)

## 1. Answer first
The final model is a tuned, regularised logistic regression (class_weight="balanced", C=0.0101) on
customer attributes (contract type, tenure, support calls, billing, usage, demographics, region).
On the untouched test set (1,200 customers, scored once): calling the top 10% ranked by the model
catches customers who actually churn at a rate of **25.8%** (95% CI 19.2%-32.5%), **about 3x**
the base churn rate of 8.7% - i.e. the retention team's 10%-of-base call list would contain roughly
3x as many true churners as calling at random, and would reach about 30% of all churners that month
(recall@10% = 29.8%). This is a real, non-leaky signal (survives a shuffled-label sanity check) but
it beats the success threshold (ASSUMED >=3x lift, i.e. ~26%+ precision) only marginally - the
lower end of the confidence interval falls short. Ship it as a ranked call list, not as a
calibrated probability (see Limitations), and treat the ASSUMED threshold as one to confirm with
the business before declaring success.

## 2. Problem
See problem_statement.md. Unit = one customer row. Target = `churned` (binary). Decision fed =
retention team calls the top-10%-by-risk each month. Primary metric = precision@10% (capacity-
limited ranking). 6 ASSUMED lines open (user unavailable for the whole 25-minute run): the exact
target window/prediction-moment timing, the success threshold, and problem constraints
(interpretability, latency) were all assumed from context rather than confirmed.

## 3. Data
6,150 raw rows / 15 columns -> 6,000 rows after cleaning. Problems found, with counts:
- **150 exact duplicate rows** (dropped).
- **contract_type: 11 spelling/case variants** of 3 real levels (normalized).
- **total_charges: 360 rows (9.0%) used a -999 sentinel** for missing (converted to NaN).
- **age: 32 rows (0.5%) had impossible values** (>120; one raw value was literally 250) (converted to NaN).
- **total_charges missing (post-sentinel): 9.95%**; **last_invoice_amount missing: 18.0%**.
- **Two leaked columns, dropped**: `churn_reason` (leak_screen.py score 1.00/1.00 - populated only
  for churned customers, i.e. written after the outcome) and `account_status_code` (score 0.99 -
  the 'C' status co-occurs with churned=1 93% of the time, read as a post-cancellation account
  state, not a pre-churn attribute). Both would have made the model look "too good to be true" if kept.
- `last_invoice_amount` was investigated as a possible third leak (higher missingness for churned
  customers, 30.5% vs 16.4%) but kept as a legitimate feature (ASSUMED) - its value tracks
  `monthly_charges` closely in both classes, with no sign of an inflated final/termination bill.
- `avg_daily_usage_mb` has a heavy right tail (max 73x the 99th percentile) - kept as-is (not
  confirmed as an error); winsorizing it made no measurable difference to the screened score.
- Adversarial validation (train vs test): ROC-AUC 0.49 (as expected for the random stratified split).

## 4. Validation design
Stratified 65/15/20 split (train 3,900 / holdout 900 / test 1,200) on `customer_id` (unique key
after dedup), 5-fold CV on train, seed 42. Chosen over a time-based split because churn rate by
signup-quarter cohort is flat (6.8%-10.9%, no trend, 2023 Q1 - 2025 Q2) and there is no repeating
entity to group on. Test-set precision@10% has an estimated 95% interval of roughly +/-0.08-0.09
(n=120 flagged rows), which is why model choice used CV, not the test set. Noise floor
(precision@10%, RepeatedStratifiedKFold 5x3, reference linear pipeline): **repeat_sd = 0.0133**.

## 5. Decisions
One line per stage (full reasoning in decision_log.md):
- Stage 1: dedup, sentinel/impossible-value cleaning, spelling normalization (stateless, fixed).
- Stage 2: stratified split chosen over time/group; noise floor 0.0133.
- Stage 3: dropped `churn_reason` and `account_status_code` as leaks; kept `last_invoice_amount` (ASSUMED legit).
- Stage 4-6: median+indicator imputation and "no outlier treatment" both tied with their alternatives within noise; no new engineered features screened (time budget).
- Stage 7: linear beats LightGBM and CatBoost by more than the noise floor; problem verdict LEARNABLE; class_weight="balanced" chosen over resampling.
- Stage 8: 30-trial Optuna joint search confirms tuned logistic regression (C=0.0101) as the outright winner; holdout precision@10% = 0.2667, consistent with the CV estimate (no overfitting to the search).
- Stage 9: test precision@10% = 0.2583 [0.1917, 0.3250]; shuffled-label sanity check passed.
- Stage 10: production pipeline includes Stage 1 cleaning as its first step; verified on raw rows.

## 6. Results
| Model / pipeline | CV metric (mean +/- sd) | Locked holdout | Test (95% CI) |
|---|---|---|---|
| Trivial baseline (prior) | 0.0846 +/- 0.0295 | - | base rate 0.0867 |
| Business rule (month-to-month + support calls) | 0.1897 +/- 0.0263 | - | - |
| Simple logistic regression | 0.2103 +/- 0.0358 | - | - |
| LightGBM (tuned, Stage 8) | ~0.2103 (best trial) | - | - |
| **Final pipeline (tuned logistic, Stage 8 winner)** | **0.2385 +/- 0.0266** | **0.2667** | **0.2583 [0.1917, 0.3250]** |

Secondary test metrics: recall@10% 0.2981 [0.2212, 0.3750]; average precision 0.2140 [0.1693, 0.2916];
ROC-AUC 0.7186 [0.6695, 0.7702]; ECE 0.3699; Brier 0.2226.

## 7. What drives the prediction
Standardized logistic-regression coefficients (feature_importance.csv), largest first:
1. **Contract type is the single strongest driver.** Month-to-month contracts carry the highest
   risk (coef +0.55); one-year and two-year contracts are protective (-0.34, -0.21).
2. **Tenure is protective** (coef -0.33): the longer a customer has been signed up, the lower the
   predicted churn risk.
3. **Support calls in the last 90 days raise risk** (coef +0.30): more contact with support tracks
   with higher churn probability.
Smaller effects: paying by electronic check (+0.21), a few regions, `last_invoice_amount` and its
missingness indicator (mildly positive), and higher `avg_daily_usage_mb` (mildly protective -
engaged users churn less). All directions match domain expectations for telecom-style churn
(month-to-month/new/high-complaint customers churn more), which is a useful sanity check given
that account_status_code and churn_reason had to be removed as leaks.

## 8. Where it fails
- **Long-tenured customers (37+ months): precision@10%-within-slice only 0.10**, vs 0.35 for
  customers in their first year - the model is much better at catching early-lifecycle churn than
  late-lifecycle churn.
- **Long contracts (one-year/two-year): precision@10%-within-slice 0.07-0.08**, close to those
  slices' own base rates - the model has little to say once a customer is not on a month-to-month plan.
- **Calibration is poor** (ECE 0.37, Brier 0.22) because `class_weight="balanced"` was used to
  handle the 8.7% positive rate; scores rank well (ROC-AUC 0.72) but should not be read as true
  probabilities without recalibration.
- Region-level slices are too small (n~43-71 each) to certify any single region as a failure mode
  beyond sampling noise; none stood out reliably.

## 9. Limitations and risks
- **6 ASSUMED lines in problem_statement.md were never confirmed with the user** (unavailable):
  the target's exact time window, the true prediction-moment timestamp, and the "3x lift" success
  threshold in particular should be checked before this is treated as a pass/fail verdict.
- `last_invoice_amount`'s recording time relative to churn is inferred from its distribution, not
  confirmed with a data owner; if it is in fact a post-cancellation final bill, it should be dropped
  and the model re-evaluated (likely a modest drop in precision@10%, since its coefficient is small).
- Probabilities are not calibrated; only the ranking (top-10% selection) has been validated.
- The 25-minute budget forced compact mode throughout: single 5-fold CV (mostly no repeats), 2
  preprocessing candidates per stage, a 30-trial joint search, only 2 tree-based families
  (LightGBM, CatBoost) and no tabular foundation model (TabPFN/TabICL, not installed) or deeper
  feature engineering.

## 10. Next steps (ranked by expected gain)
1. **Confirm the ASSUMED lines with the retention team** - especially the target window, true
   scoring cadence, and whether 3x lift is actually "good enough" to fund the calling program.
2. **Get billing-history depth / usage trend / complaint-text features** if available - Stage 7's
   learning curve showed the current feature set is a real but partial signal (linear model wins,
   trees overfit), so more informative features likely help more than more modeling time.
3. **Confirm `last_invoice_amount` timing with the data owner**; drop it if it turns out to be
   post-cancellation.
4. **Recalibrate probabilities** (e.g. `CalibratedClassifierCV`) if the team ever wants raw churn
   probabilities rather than just a ranked call list.
5. Re-run Stage 7/8 with a larger trial budget and, if the licence/hosted-inference questions are
   answered, a tabular foundation model (TabPFN/TabICL) given the small row count (<=5,000-row
   band in ds-model-selection).
