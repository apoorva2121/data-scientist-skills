# Churn propensity model: report
Status: Stages 0-10 all passed. Final pipeline saved (`final_pipeline.skops`).

## 1. Answer first
The final model is a regularized logistic regression on 8 honest features (contract type, tenure,
billing, support calls, usage, payment method, age). On the untouched test set (n=1200, 104
churners) it scores **PR-AUC 0.211 (95% CI 0.166-0.288)** and **ROC-AUC 0.716 (0.666-0.767)**.
Used the way the retention team will use it - call the top 10% by score each month - it delivers
**precision@10% = 22.5% (16.4-30.0%)** and **recall@10% = 26.0% (19.2-34.6%)**, i.e. about 1 in 4-5
of the people called would actually have churned (versus 8.7% if calls were random - roughly a
2.6x lift), and the calls would catch about a quarter of that month's churners. This beats a
free hand-built business rule (precision@10%=20.7%, recall@10%=27.9%) directionally but not
decisively on this one test set - their confidence intervals overlap. It clears the trivial and
shallow-tree baselines by a wide, unambiguous margin. The three things that actually drive churn
in this data are **contract type** (month-to-month vs. a term contract), **support-call frequency**,
and **tenure** - in that order, agreeing across three independent checks (model coefficients,
permutation importance, raw EDA). Four boosted-tree families (LightGBM, XGBoost, CatBoost, sklearn
HistGBM) were screened and tuned and all underperformed logistic regression on this data - not a
modelling gap, the signal here is close to linear and the positive count (338 train positives)
favors the simpler, lower-variance family. The Stage 0 success threshold (precision@10% >= 3x base
rate, i.e. >=26%) is **not clearly met** on this test set (point estimate 22.5%, CI spans values
above and below 26%) - reported honestly rather than rounded up.

## 2. Problem
Predict which customers will churn so the retention team can call the highest-risk ~10% each
month. Framed as capacity-constrained ranking, not plain classification (`problem_statement.md`).
Open ASSUMED lines never confirmed with the user/data owner: single-snapshot cross-section (no
true as-of date in the source file); prediction moment = "now"; the precision@10% >= 3x base rate
success threshold (no real call-cost/save-rate economics were available); no
latency/fairness/regulatory constraint; monthly retrain cadence.

## 3. Data
6,150 raw rows, 15 columns, 1 target (`churned`, 8.65% positive). Cleaning (Stage 1): 150 exact
duplicate rows dropped (-> 6,000 rows); `total_charges` -999 sentinel -> NaN (360 rows, 10.0%
missing combined with pre-existing NaNs); `age` impossible values {0, 250} -> NaN (60 rows, 1.0%);
`contract_type` spelling/case/whitespace (12 raw variants) -> 3 canonical levels. Leak hunt (Stage
3, train-only): `churn_reason` (filled only when churned=1, leak_screen score 1.00) and
`account_status_code` (status='C' co-occurs with churned=1 in 93% of churned rows, score 0.9949)
are both **post-outcome fields and were dropped** - the single biggest reason a naive model on
this data would look "too good to be true." `last_invoice_amount` was kept as an open risk through
Stage 4 (18% missing, informative missingness, 0.97 Spearman-redundant with `monthly_charges`) and
dropped at Stage 6 once an ablation showed zero benefit. Outliers (Stage 5): `avg_daily_usage_mb`
has a genuine heavy tail (15 rows >10x the 99th percentile, manually verified as not data-entry
errors); no treatment beat "none" beyond noise, matching this exact evaluation set's documented
behaviour in `ds-outliers/SKILL.md`.

## 4. Validation design
Stratified train(3,900)/holdout(900)/test(1,200) split by `customer_id` (no time index trustworthy
enough to split on; no repeating entity to group by). 5-fold CV on train for all screening.
Adversarial validation train-vs-test AUC = 0.494 (healthy). Noise floor (fold_sd) = 0.042 on
PR-AUC - the yardstick every later comparison was checked against. Test-set precision estimate
pre-registered at Stage 2 (~+/-0.08 on precision@10%) matched what Stage 9 actually observed.

## 5. Decisions
Full reasoning for every stage in `decision_log.md` (10 stage blocks, each with considered /
applied / carried forward / skipped / evidence / open risks). Headline decisions: drop
`churn_reason` + `account_status_code` (leak); drop `last_invoice_amount` + `region` (Stage 6
ablation, no gain); drop 3 hand-built engineered features (interaction, frequent-caller flag,
billing residual - none cleared the noise floor); `class_weight="balanced"` for imbalance
(SMOTE screened, no better); logistic regression over 4 boosted-tree families (all screened,
tuned, still behind); tie-break rule ("simpler wins within noise") applied at Stages 5, 6 and 8.

## 6. Results
| Model / pipeline | CV metric (PR-AUC, mean +/- sd) | Locked holdout (PR-AUC) | Test PR-AUC (95% CI) | Test precision@10% (95% CI) |
|---|---|---|---|---|
| Trivial baseline (dummy) | 0.087 +/- 0.001 | - | 0.087 (0.087, 0.087) | 0.087 (degenerate*) |
| Business rule | 0.178 +/- 0.028 | - | 0.192 (0.149, 0.263) | 0.207 (0.153, 0.268) |
| Shallow tree (depth 3) | 0.145 +/- 0.014 | - | 0.172 (0.142, 0.212) | 0.233 (0.175, 0.293) |
| Best boosted tree (tuned CatBoost) | 0.194 +/- 0.032 | - | not carried to final (see Stage 8) | - |
| **Final pipeline (logistic regression)** | **0.217 +/- 0.040** | **0.209** | **0.211 (0.166, 0.288)** | **0.225 (0.164, 0.300)** |

\*dummy's precision@10% is degenerate: a constant score makes the "top 10%" cut ill-defined; shown
for completeness only, not a real comparison point.

CV mean (0.217), holdout (0.209) and test (0.211) all agree within ~0.01 - no leakage inflation.
Test ROC-AUC (0.716, CI 0.666-0.767) lands almost exactly on this dataset's documented honest
ceiling (0.722, CI 0.671-0.774, per `ds-evaluation/SKILL.md`). Shuffled-label sanity check: PR-AUC
falls from 0.217 (real labels) to 0.101 (shuffled) against an 0.087 base rate - matches that same
reference almost exactly (0.099 vs 0.087) - strong independent confirmation nothing is leaking.

## 7. What drives the prediction
Three independent methods agree on the same top 3, in the same order:
1. **Contract type** - month-to-month customers are far higher risk than one/two-year customers
   (standardized coefficient +0.65 for month-to-month vs -0.39/-0.25 for one/two-year;
   permutation-importance PR-AUC drop 0.062, the largest of any feature).
2. **Support calls in the last 90 days** - more calls, more risk, monotonically (7.0% churn at
   0-1 calls rising to 21.6% at 4-7 calls in raw EDA; coefficient +0.33; permutation drop 0.060).
3. **Tenure** - longer-tenured customers are markedly less likely to churn (13.9% churn in the
   first 8 months falling to 4.7% at 60-72 months; coefficient -0.39; permutation drop 0.049).

Everything else (monthly charges, usage, payment method, age, total charges) is a distant second
tier, each under 0.005 permutation-importance drop - essentially noise-level on this data.
None of the top drivers are on the Stage 3 leak table - all are legitimately available before the
churn event, so this is a defensible, not-too-good-to-be-true answer to "what actually drives churn."

## 8. Where it fails
Weakest slices: one-year-contract customers (test PR-AUC 0.077, n=283) and long-tenure
49-72-month customers (0.120, n=372) - both have a very low base rate in-segment, which makes
ranking inherently harder, not a sign the model is broken. Region slices span 0.04-0.55 across 20
levels with wide, overlapping intervals at n=43-71 each - confirms the Stage 6 decision to drop
region (mostly sampling noise). Calibration is poor (ECE 0.372, Brier 0.226) - class weighting
inflates predicted probabilities; fine for ranking (the deployment use), not for reporting raw
"X% chance of churning" numbers. Error analysis: missed churners (false negatives) are
disproportionately long-tenure, annual-contract customers with no recent support calls - churns
with none of this dataset's known risk signals, plausibly driven by factors not captured here
(price shock, a competitor offer, a life event). False alarms are almost all short-tenure,
month-to-month customers with several recent support calls - textbook at-risk profiles that
happened not to churn in this window; likely still worth a call.

## 9. Limitations and risks
See `model_card.md` for the full list. Headline items: (1) the Stage 0 success threshold and
program economics (call cost, save rate, customer LTV) were never confirmed with the retention
team - this report cannot say the program pays for itself, only that the ranking beats random and
probably beats a free heuristic; (2) no true as-of timestamp exists in the source data, so
"prediction moment = now" is an assumption; (3) calibration is poor - do not surface raw
probabilities as real percentages; (4) the model is weakest for long-tenure/annual-contract
customers - do not expect it to catch every at-risk long-tenure account; (5) `last_invoice_amount`'s
true recording time relative to churn was never confirmed with a data owner - moot for this model
(it is excluded) but relevant if reconsidered later.

## 10. Next steps (ranked by expected gain)
1. **Confirm program economics with the retention team** (call cost, save rate, LTV) so
   precision@10% can be translated into a dollar answer and the success threshold set for real,
   rather than the ASSUMED 3x-base-rate placeholder used here.
2. **Get a larger or longer-history test set.** 104 test positives gives a real but wide interval
   (+/-0.08 on precision@10%); a few more months of data would sharpen every claim in this report,
   including whether the model truly beats the business rule.
3. **Add features the dataset does not have**: competitor pricing/offers, a satisfaction survey,
   a recorded reason for contract changes - the false-negative analysis (Section 8) suggests the
   model's ceiling is capped by missing information, not by model choice.
4. **Recalibrate** (isotonic on a fresh holdout) if raw probabilities are ever needed downstream.
5. **Build the `pandera` schema and drift monitors** named in `monitoring_plan.md` - not built in
   this iteration given the time budget.
6. Consider TabPFN/TabICL (recommended by `ds-model-selection` for datasets this size) if the
   environment gets these packages installed - not evaluated here since they are not installed
   and installing packages was out of scope for this run.

## Where the skills were unclear, wrong, or hard to follow
- `ds-model-selection/SKILL.md` recommends TabPFN-3.5/TabICLv2 "up to about 5,000 rows"; this
  project has 3,900 train rows, right at that boundary, but neither package is installed and the
  environment note says not to install packages - the skill has no explicit fallback for
  "recommended family unavailable in this environment," so it was treated the same as a hardware
  gate (state the requirement, don't silently drop it, don't silently install it) and skipped
  with a logged reason instead. A one-line fallback rule would help.
- `ds-method-comparison/scripts/screen.py`'s `summarize()` marks a candidate `keep=True` whenever
  it is within the **larger** of its own corrected SE or the Stage-2 `noise_floor_sd`. With only
  5 CV folds the corrected SE is itself noisy (the script's own warning says so), and using the
  wider Stage-2 floor as a floor is reasonable, but it meant every finalist in the Stage 8 table
  (including CatBoost, 0.03 behind logistic regression) came back `keep=True` - the automatic
  1-SE-and-simplicity rule still picked correctly here, but a reader relying on the `keep` column
  alone could easily under-read how much weaker the boosted trees actually were. Worth a note in
  the skill that `keep=True` at 5 folds is a weak signal, not a strong one.
- Everything else in the eight step-skills and the orchestrating `data-scientist/SKILL.md`
  followed cleanly; the worked "on the churn evaluation set..." examples embedded in several
  skills (ds-model-selection, ds-outliers, ds-method-comparison, ds-evaluation) turned out to be
  direct, checkable predictions for this exact dataset and matched what was independently found
  here (logistic regression beating boosting, no outlier treatment beating "none", ROC-AUC ~0.72,
  shuffled-label PR-AUC ~0.10) - a strong, reassuring consistency check on the whole pipeline.
