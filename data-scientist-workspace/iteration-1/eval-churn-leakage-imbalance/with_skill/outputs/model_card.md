# Model card: churn-propensity-logreg v1
Date: 2026-09-27   Owner: data science (this project)   Contact: retention-analytics channel (ASSUMED)

## Intended use
- Decision supported: the retention team ranks the current customer base once a month and calls
  the top ~10% by predicted churn probability. The model supports a **ranking/triage** decision,
  not an automated action - a human still decides what to say on the call.
- In scope: existing customers, snapshot scoring (see Limitations: no explicit as-of date exists
  in the source data), the 8 raw account/billing/usage/demographic fields listed below.
- Out of scope: new/prospective customers (no tenure yet), any use of the raw output as a
  calibrated probability (see Performance - calibration), any decision with legal/financial
  stakes beyond "who gets a retention call."

## Model
- Pipeline: RawRowCleaner (sentinel -999->NaN, impossible age {0,250}->NaN, contract_type
  case/whitespace normalization - the Stage 1 stateless rules, reapplied inside the pipeline so
  it is safe on raw production rows) -> median imputation (total_charges, age) -> StandardScaler
  (numeric) + OneHotEncoder (contract_type, payment_method) -> LogisticRegression(C=0.01915,
  class_weight="balanced", max_iter=3000).
- Features (8): tenure_months, monthly_charges, total_charges, support_calls_90d,
  avg_daily_usage_mb, age, contract_type, payment_method. Explicitly excluded: churn_reason and
  account_status_code (post-outcome leaks, Stage 3), last_invoice_amount and region (dropped on
  Stage 6 ablation evidence - no measurable gain, and each carried its own risk: last_invoice_amount
  a redundancy/leak-timing risk, region a noisy 20-level categorical).
- Training data: churn.csv snapshot, 6,150 raw rows / 6,000 after deduplication, 8.67% churned.
  Train+holdout used for the final fit = 4,800 rows (headline test set: 1,200 rows, untouched
  until Stage 9 scored it once).
- Library versions: see `model_versions.json` (scikit-learn 1.7.2, pandas 2.3.x, numpy 2.3.x,
  skops 0.15.0, seed=0 throughout). Saved artifact: `final_pipeline.skops`.

## Performance (untouched test set, n=1200, 104 positives, 95% bootstrap intervals)
| Metric | Model | Business rule | Simple baseline (dummy) |
|---|---|---|---|
| PR-AUC (average precision) | 0.211 (0.166, 0.288) | 0.192 (0.149, 0.263) | 0.087 (floor) |
| ROC-AUC | 0.716 (0.666, 0.767) | 0.692 (0.639, 0.747) | 0.500 |
| Precision @ top 10% called | 0.225 (0.164, 0.300) | 0.207 (0.153, 0.268) | 0.087 (degenerate, see note) |
| Recall @ top 10% called | 0.260 (0.192, 0.346) | 0.279 (0.192, 0.385) | 1.000 (degenerate, see note) |

Threshold: capacity rule, not a probability cutoff - each month, score every customer and call the
top 10% by score (pre-registered on the holdout: score cutoff 0.676, holdout precision@10%=22.2%,
recall@10%=25.6%, closely matching the test numbers above).
Calibration: **poor** - ECE=0.372, Brier=0.226 (`class_weight="balanced"` inflates predicted
probabilities for the minority class; reliability table in `calibration_table.csv`). Ranking is
unaffected by this (monotonic), but the raw probability number should not be shown to users as
"X% chance of churning" without first fitting an isotonic calibrator on the holdout.
Honest-ceiling check: test PR-AUC (0.211), holdout (0.209) and CV (0.217) all agree within ~0.01;
test ROC-AUC (0.716) sits almost exactly on this dataset's documented honest ceiling (see
`ds-evaluation/SKILL.md`: 0.722, CI 0.671-0.774) - consistent with no leakage surviving into the
final model. On this one 1200-row test set the precision/recall@10% edge over the free business
rule is directionally consistent with every earlier CV/holdout screen but its confidence intervals
overlap the business rule's - read this as "probably meaningfully better," not a decisive proof.

## Performance by segment
Weakest named slices: one-year-contract customers (PR-AUC 0.077, n=283, very low base rate makes
ranking inherently hard) and long-tenure 49-72-month customers (PR-AUC 0.120, n=372, same reason).
Region slices span 0.04-0.55 PR-AUC across 20 levels at n=43-71 each with wide, overlapping
intervals - noise, not a real regional effect (this is why region was dropped as a feature).
Full table: `slice_table.csv`.

## Fairness
No protected attribute (race, gender, disability, etc.) exists in this dataset; `age` is present
but not a legally protected class in most retention-call contexts (ASSUMED - confirm with legal/
compliance before any regulated use). Not assessed with `fairlearn` given no protected column to
group by; region and payment_method were checked as ordinary business slices only, not a fairness
audit. If a protected attribute becomes available, a fairness pass should be added before scaling
this beyond an internal pilot.

## Explanations
Top 3 drivers, agreeing across standardized coefficients, test-set permutation importance and the
Stage 3 EDA: **contract_type** (month-to-month raises risk, one/two-year contracts lower it -
permutation drop 0.062), **support_calls_90d** (more calls, more risk - permutation drop 0.060),
**tenure_months** (longer tenure, less risk - permutation drop 0.049). Everything else
(monthly_charges, avg_daily_usage_mb, payment_method, age, total_charges) is a distant second tier,
each under 0.005 permutation drop - essentially noise-level on this data. None of the top features
are on the Stage 3 leak table as leak-suspects. Full tables: `model_coefficients.csv`,
`permutation_importance.csv`.

## Limitations and risks
- ASSUMED: single-snapshot cross-section (no true as-of timestamp in the source data); prediction
  moment = "now"; success threshold (precision@10% >= 3x base rate) was never confirmed with the
  retention team's real economics (call cost x save rate x customer LTV) - **confirm before scaling
  spend on this program**.
- last_invoice_amount's true recording time relative to churn was never confirmed with a data
  owner (no data dictionary available) - it is not used as a feature, so this risk is now moot for
  this model, but would matter again if it were reintroduced later.
- Calibration is poor (see above) - do not surface raw probabilities as calibrated risk percentages.
- The model is weakest exactly where churn is rarest (long tenure, annual contracts) - the
  retention team should not expect it to find every at-risk long-tenure customer.
- Missed churners (false negatives) are disproportionately annual-contract, no-recent-support-call
  customers - this model cannot see whatever actually drove them to leave (price shock, a
  competitor offer, a life event are all plausible but not in this dataset).
- The skops-saved artifact's custom `RawRowCleaner` step is an "untrusted type" on reload by
  design (skops flags any class not from a vetted library) - only load `final_pipeline.skops`
  from this project's own trusted code path, never from an unknown source, and move
  `RawRowCleaner` into an importable module (not a throwaway script) before deploying to a
  different process.

## Monitoring and maintenance
See `monitoring_plan.md`.
