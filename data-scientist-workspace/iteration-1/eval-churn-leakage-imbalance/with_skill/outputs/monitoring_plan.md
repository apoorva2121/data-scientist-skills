# Monitoring plan - churn propensity model v1

## Drift checks (monthly, before each scoring run)
- Per-feature drift on the 8 model features: PSI or KS test of this month's distribution vs the
  training distribution. Watch `contract_type` mix, `support_calls_90d` distribution and
  `tenure_months` distribution first - the top 3 drivers, so drift there directly threatens the
  ranking.
- Prediction-distribution drift: compare this month's score distribution (mean, deciles) to the
  training-time distribution. A shift in the score needed to reach the top 10% is expected as the
  customer base changes seasonally; a *sudden* jump is a signal to investigate before calling.
- Schema/data validation on each incoming batch: same 8 columns present, contract_type in
  {month-to-month, one year, two year} after normalization, no unexpected new sentinel values
  (e.g. a new "-999"-style code introduced upstream). A `pandera` schema mirroring
  `column_audit.csv`'s verdicts is recommended (not built in this iteration - time-boxed out).

## Performance tracking (delayed-label)
- True churn labels arrive with a lag (a customer's status a month later). Once available,
  recompute precision@10%/recall@10% for that month's calls and compare to the test-set numbers
  in `model_card.md` (P@10%=22.5%, R@10%=26.0%). A sustained drop of more than ~1 test-set standard
  error (~0.07 on precision@10% at this call volume) for two consecutive months is the trigger below.
- Track the business outcome too, not just the model metric: of the customers called, how many
  were actually retained (needs a save-rate field this project did not have) - this is the number
  that ultimately justifies the program, not PR-AUC.

## Retraining trigger
- Retrain when either: (a) two consecutive months show precision@10% or recall@10% down by more
  than one test-set standard error from the numbers in `model_card.md`, or (b) PSI > 0.2 on any of
  the top-3 driver features, or (c) 6 months have passed since the last fit (ASSUMED cadence,
  confirm with the retention team) - whichever comes first.
- On retraining, rerun this project's Stage 1-9 pipeline on fresh data rather than only refitting
  the same features/hyperparameters - re-check the leak table (Stage 3) every time, since a new
  data extract could reintroduce a churn_reason/account_status_code-style field or a new one.

## Data validation
- Recommended schema file: derive a `pandera` schema from `column_audit.csv` (types, plausible
  ranges: age 18-85, tenure_months 1-72+, support_calls_90d >=0, contract_type in the 3 canonical
  levels) and run it on every incoming batch before scoring. Not built in this iteration - flagged
  as the top "next step" in `report.md`.

## Known open risk carried into monitoring
Region and last_invoice_amount are not used by the model but still exist in the raw feed; if a
future iteration reconsiders adding them back, treat that as a new Stage 6 decision with fresh
ablation evidence, not a default re-inclusion.
