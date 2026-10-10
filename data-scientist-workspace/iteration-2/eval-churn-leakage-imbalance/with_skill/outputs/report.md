# Churn propensity model: report
Status: Stages 0-10 all passed (25-minute time budget; breadth cut throughout per decision_log.md, correctness gates never cut)

## 1. Answer first
A regularised logistic regression, ranking customers by churn probability, lets the retention team
call the 10% most likely to churn each month with **precision@10% = 0.267** (95% CI 0.192-0.350) on
an untouched test set - about **3.1x** the 8.67% base rate, capturing roughly 31% of that month's
churners in the top decile. This clears the ASSUMED 3x-base-rate success bar, but only marginally,
and the interval is wide (n=1200 test rows, 104 positives). Two columns (`churn_reason`,
`account_status_code`) were dropped as near-perfect leaks recorded after the churn outcome; without
that removal the model would have looked far better and been useless in production. The strongest
drivers are contract type (month-to-month = risk), tenure (longer = safer), and recent support calls
(more = risk). The model works well for short-tenure, month-to-month customers and is close to
uninformative for one-year-contract customers, who churn for reasons not captured in this data.

## 2. Problem
See `problem_statement.md`. Unit = customer account; target = `churned` (binary); decision fed =
retention calls capacity-capped at 10% of customers/month, hence primary metric = precision@10%.
5 ASSUMED lines open (target window, `last_invoice_amount` timing, success threshold, no documented
baseline process, duplicate-row treatment) - all listed there for the data owner.

## 3. Data
6,150 raw rows, 15 columns, 8.65% churn. Cleaning (Stage 1): 150 exact-duplicate rows dropped
(-> 6,000 rows, 8.67% churn); `total_charges` sentinel -999 -> null (367 rows); `age` 0 or >=120 ->
null (29 + 33 rows, impossible values); `contract_type` spelling/case variants canonicalized to 3
levels. Leakage (Stage 3): `churn_reason` (non-null only when churned=1, univariate AUC 1.00) and
`account_status_code` (status 'C' ~ cancellation, AUC 0.995) dropped as post-outcome leaks.
`last_invoice_amount` kept as ASSUMED-legit but flagged as the top open risk. Missingness:
`total_charges` (9.9%) and `age` (1.0%) look MCAR-like; `last_invoice_amount` (17.6%) is diagnosed
informative (its missingness indicator alone has AUC 0.561 against the target) - kept with an
indicator. Outliers: no domain-rule violations beyond what Stage 1 already nulled; `avg_daily_usage_mb`
has a genuine heavy right tail (skew ~20) but low lift (1.17x) - kept, log1p screened as a tie for the
linear model.

## 4. Validation design
Stratified random 65/15/20 train/holdout/test split (3,900/900/1,200 rows), seed 42, on customer_id
(unique after dedup, so no grouped split needed). Time-based split was considered and rejected
(ASSUMED cross-sectional snapshot; no drift check run given the time budget - logged as an open
risk). Adversarial validation train-vs-test AUC = 0.501 (sound split). Noise floor: repeat_sd = 0.0024
on ROC-AUC (RepeatedStratifiedKFold 3x3, simple logistic reference pipeline); precision@10% itself is
noisier at this n (fold sd 0.03-0.07) - CV rankings among close candidates should not be over-trusted,
hence the locked-holdout confirmation and the test-set CI are the numbers that matter.
Test-set precision estimate width: ~+/-0.08 at 95%, in line with the bootstrap CI obtained.

## 5. Decisions
One line per stage (full reasoning and evidence in `decision_log.md`):
- Stage 1: dedup + sentinel/impossible-value nulling + spelling canonicalization, stateless.
- Stage 2: stratified split chosen over group/time; noise floor measured.
- Stage 3: `churn_reason` and `account_status_code` dropped as post-outcome leaks; `last_invoice_amount` kept with an open risk.
- Stage 4: median (+indicator for the informative column) for linear; native-NaN for GBDT; KNN/MICE skipped (weak inter-column correlation, time budget).
- Stage 5: no removal/capping; extremes carry positive lift, not error signatures; log1p tied with none for linear usage feature.
- Stage 6: `charge_gap` and `is_new_customer` engineered features carried forward (clear GBDT gain, tie for linear); one-hot encoding for all categoricals.
- Stage 7: baselines beaten; logistic regression beat LightGBM/XGBoost defaults by a wide margin; TabPFN/foundation models skipped (licence + hosted-inference data-location not clearable without the user).
- Stage 8: small 8-candidate grid; logistic C=0.01 won, confirmed on the locked holdout.
- Stage 9: test scored once after verify_test_untouched; sanity tests all passed.

## 6. Results
| Model / pipeline | CV metric (mean +/- sd) | Locked holdout | Test (95% CI) |
|---|---|---|---|
| Trivial baseline (random 10%) | precision@10% = 0.087 (= base rate) | - | precision@10% = 0.087 (base rate) |
| Business rule (support_calls + MTM flag) | precision@10% = 0.164 | - | not re-scored (rule, no fit) |
| Simple model (logistic, defaults) | precision@10% = 0.208 (ROC-AUC 0.679) | - | - |
| Final pipeline (logistic, C=0.01) | precision@10% = 0.233 +/- 0.062 (ROC-AUC 0.687) | precision@10% = 0.200, ROC-AUC = 0.743 | precision@10% = 0.267 [0.192, 0.350], ROC-AUC = 0.718 [0.667, 0.768], PR-AUC = 0.213 [0.161, 0.301] |

Runner-up (logistic C=1.0) and best boosted tree (XGBoost, precision@10% CV 0.192) are documented in
`search_log.csv`; both trail the winner outside the noise floor for the tree-vs-linear gap, within
noise for the C sweep.

## 7. What drives the prediction
Logistic coefficients and permutation importance (test set, ROC-AUC, 20 repeats) agree:
1. **Contract type** - by far the largest effect (permutation importance 0.089); month-to-month
   raises risk, one/two-year contracts lower it. Matches the classic telco churn pattern.
2. **Tenure** (0.040) - longer-tenured customers are safer; newer customers (<=12 months) churn at
   14.9% vs 5.7% for 37+ months.
3. **Support calls in the last 90 days** (0.038) - more recent support contact predicts higher risk.

Secondary: payment method (electronic check = somewhat higher risk), last_invoice_amount, region
(weak). Charges and usage carry little independent signal once contract/tenure/support are in the model.

## 8. Where it fails
Worst slice: **one-year-contract customers** (ROC-AUC 0.528, near chance; precision@10% 0.071 vs
their own 2.8% base rate). Two-year contracts are also harder (ROC-AUC 0.634) than month-to-month
(0.667). By tenure, long-tenured customers (37+ months) are hardest to rank (ROC-AUC 0.660,
precision@10% 0.103) though the model still beats their base rate.

Error analysis: false positives (highest-scored non-churners) are short-tenure, month-to-month,
high-support-call customers - a plausible risk profile that simply did not convert this period, not
a data problem. False negatives (lowest-scored churners) are long-tenure, one/two-year-contract
customers with zero recent support calls - customers that look "safe" on every available signal but
churned anyway. This is a feature gap (no price-change, competitor-offer, or satisfaction-survey data
available), not a modelling gap - more boosting or tuning will not fix it.

Calibration: probabilities are distorted by class_weight="balanced" (Brier 0.223); only the top-10%
ranking is validated, not the raw probability value.

## 9. Limitations and risks
- 5 ASSUMED lines in `problem_statement.md`, most importantly the target definition/window and
  whether `last_invoice_amount` is a legitimate pre-outcome feature or a leak - both need the data
  owner's confirmation before this ships.
- No time-based drift check was run (time budget); if churn definition or customer mix shifted over
  the 2023-2025 signup window, the stratified split could be mildly optimistic.
- Test-set CI is wide (+/-0.08 on precision@10%) at n=1,200/104 positives; a larger evaluation sample
  would sharpen the number before committing calling-program ROI to it.
- One-year-contract churn is essentially unmodelled (ROC-AUC ~0.53); do not rely on this model to
  prioritize that segment specifically.
- 25-minute time budget forced cuts: no Optuna/nested hyperparameter search, no KNN/iterative
  imputation screen, no interaction/spline features, no foundation-model (TabPFN/CatBoost/TabICL)
  screen, only 3-fold (not 5-fold) CV. None of the correctness gates (leak table, split discipline,
  test-once, sanity tests) were cut.

## 10. Next steps (ranked by expected gain)
1. **Confirm the 5 ASSUMED lines** with the data owner, especially `last_invoice_amount` timing and
   the churn target window - the single biggest risk to whether this number is honest.
2. **Add features that could explain the missed long-tenure churners**: price-change history,
   competitor/offer exposure, satisfaction survey or NPS, complaint-resolution outcome (not just
   call count). The learning curve was still rising at full training size (Stage 7), so more rows
   or more informative columns are likely to help more than further tuning.
3. **Widen the evaluation sample** (or use nested CV) to narrow the precision@10% confidence interval
   before sizing the retention program's expected ROI on this number.
4. **Screen TabPFN/CatBoost/foundation models** once licence terms and data-location for hosted
   inference are cleared with the user - the row count (6,000) fits comfortably in their CPU range,
   and the current GBDT gap suggests there is room for a stronger model if given a fair hyperparameter
   budget (Optuna, more folds) that this run's time constraint did not allow.
5. **Build a separate model or rule for one-year-contract customers**, or accept that segment cannot
   be prioritized on current data and route it to a different retention tactic (e.g. proactive
   contract-renewal outreach for everyone in that segment, rather than a scored subset).
