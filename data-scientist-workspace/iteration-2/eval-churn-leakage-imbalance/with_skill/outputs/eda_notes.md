# EDA notes (Stage 3, train rows only)

Adversarial validation (train vs test) AUC: 0.5014 - near 0.5, split looks sound.

Leak table summary: churn_reason and account_status_code both written after the churn outcome (perfect / near-perfect correspondence) -> dropped as fixed steps, never searched over.

last_invoice_amount kept with an open-risk assumption about its timing; if the data owner confirms it is a post-cancellation final invoice, it must be dropped and the model re-evaluated.

avg_daily_usage_mb is right-skewed with a long tail (max 4196 vs median ~20) -> candidate for log1p transform / outlier capping at Stage 5.

support_calls_90d and contract_type=Month-to-month show the expected direction (more calls, shorter/no-commitment contracts -> higher churn) in the univariate AUC table (leak_table.csv), consistent with domain expectations - no sign flips observed.
