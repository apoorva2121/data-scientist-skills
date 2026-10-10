# Problem Statement — Churn Propensity Model

| Field | Value |
|---|---|
| Unit of analysis | One customer (`customer_id`) at the snapshot moment the file was extracted. ASSUMED: the file is a single cross-sectional snapshot, not a panel — no snapshot date column is provided, and `signup_date` does not correlate with `tenure_months` (r=0.019), so it cannot be used to reconstruct an as-of date. |
| Target | `churned` (binary, 1 = churned). ASSUMED definition: "churned by the snapshot date," source and exact churn-event definition not documented. `churn_reason` and `account_status_code` are populated *as a consequence of* churn (see Stage 1/3 leak findings) and are not usable as predictors. |
| Prediction moment | ASSUMED: "today," i.e. the retention team scores the current active-ish customer base once a month and must use only information that would be on file *before* any cancellation event — tenure, plan, billing, usage, support history, demographics. No feature that is only populated *after* a customer leaves may be used. |
| Decision fed | Retention team calls the top-ranked customers, capacity-constrained to ~10% of the customer base per month. This is a **ranking / triage** decision, not a pure classification decision — the operating point is fixed by capacity, not by a probability threshold chosen from cost ratios. |
| Problem type | Binary classification, used downstream as a ranking (propensity-to-churn score). |
| Primary metric + reason | **PR-AUC (average precision)** as the threshold-free model-quality metric, plus **Precision@10%** and **Recall (capture-rate)@10%** as the deployment metric, because the action is capacity-capped at ~10% of customers regardless of how many are actually at risk. Accuracy is meaningless here (91% accuracy from predicting "no churn" for everyone) and full-range ROC-AUC overweights the 90% of the ranking the retention team will never act on. |
| Secondary metrics | Calibration of the score within the top decile (does "top 10%" actually contain the highest-risk customers, not just a well-ranked-but-miscalibrated score), lift/gain curve across deciles, per-segment precision (contract type, region, tenure band) to catch a model that is good on average but blind in some slice. |
| Baseline to beat | (a) Naive-rate baseline: random calling, expected precision@10% = base rate ≈ 8.7%. (b) A simple heuristic baseline: rank by `support_calls_90d` (or a small rule using tenure + contract type), which is what a retention team could do today without any model. The model must beat both by more than the noise floor to justify deployment. |
| Success threshold | ASSUMED (no target given by the user): Precision@10% at least 3x the base rate (i.e. ≳26%) and capture-rate@10% materially above the 10% a random sample would capture. Framed as ASSUMED — recommend confirming with the retention team what conversion/save-rate on a call makes a given precision worthwhile economically. |
| Constraints | ASSUMED: no strict latency constraint (monthly batch scoring, not real-time). Interpretability matters somewhat (retention team wants to know "what drives churn" to inform call scripts), so a model that also supports a simple driver explanation (SHAP / coefficients) is preferred over a total black box, but this does not rule out gradient boosting. No fairness/regulatory constraint stated. Retraining cadence ASSUMED monthly, alongside the scoring cadence. |
| Data size | 6,150 rows read; 150 exact-duplicate rows (75 customers double-counted) → 6,000 unique customers after cleaning. 14 candidate feature columns + target. Churn rate 8.65% (≈520 positives after dedup) — meaningfully imbalanced. |
| Time and compute budget | ASSUMED per environment note: total compute budget ≈20 minutes; joint hyperparameter search capped at ~60 trials / 10 minutes. No GPU. This rules out large search spaces, deep learning, and heavy stacked ensembles; favors LightGBM/XGBoost/logistic-regression-class models with a modest, budgeted Optuna search. |

**Does this need ML at all?** No. A single-rule heuristic (e.g., "call everyone with a month-to-month contract and ≥2 support calls") is checked as a baseline in Stage 7, but initial EDA already shows churn concentrated in specific segments *and* driven by a continuous risk gradient (usage, tenure, billing), so a rule alone is unlikely to hit the precision bar a ranking model can reach — this is confirmed, not assumed, at the baseline gate.

## Path
Cross-sectional binary classification path (Stages 0–9 as listed in `data-scientist/SKILL.md`); not forecasting (no reliable time index) and not unsupervised (labeled target present).

## ASSUMED lines requiring confirmation with the user / data owner
1. The snapshot is a single point in time (no repeated-measures panel).
2. "Prediction moment" = now, before any cancellation-related fields are populated.
3. Success threshold of Precision@10% ≳ 3x base rate — needs a real economic number (save rate x customer LTV x call cost) from the retention team.
4. No latency/regulatory/fairness constraints.
5. Monthly retraining cadence.
