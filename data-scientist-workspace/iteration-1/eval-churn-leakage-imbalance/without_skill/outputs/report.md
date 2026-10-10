# Customer Churn Model — Report

## 0. Summary for the retention team

- **Final model:** Logistic Regression (L2, tuned), chosen after comparing four tuned model families.
- **How good is it really:** On a held-out test set the retention team never influenced during training, calling the **top 10% highest-risk customers** (as ranked by the model) catches **~30% of all customers who actually churn**, and **~26% of the people you call will genuinely be at risk** (vs. an 8.7% base churn rate). That's roughly **3x better than calling a random 10% of customers.**
- **Top churn drivers:** (1) contract type — month-to-month customers churn far more than customers on 1- or 2-year contracts, (2) recent support call volume — customers with more support calls in the last 90 days are more likely to churn, (3) tenure — newer customers churn more than long-tenured ones.
- **Important caveat:** this dataset had serious data-quality and leakage problems (detailed below) that would have produced a fake "near-perfect" model if not caught. The number above (30% capture rate at 10% call capacity) is the honest, leakage-free estimate.

---

## 1. What was done, in order

1. **Loaded the raw file** `evals\datasets\churn.csv` (6,150 rows, 15 columns, one row nominally per customer).
2. **Explored the data** for leakage, duplicates, missingness, and invalid values before touching any model (see Section 2).
3. **Cleaned the data** (`01_prepare_data.py`): dropped exact duplicate rows, dropped two outcome-leaking columns, recoded sentinel/placeholder values to missing, normalized inconsistent category text, dropped a redundant/suspect column and identifier columns. Saved `churn_clean.csv` (6,000 rows, 10 columns).
4. **Split the data once**, up front: 80% train (4,800 rows) / 20% held-out test (1,200 rows), stratified on churn, `random_state=42`. The test set was **never used for any model-fitting, feature selection, or hyperparameter decision** — it is touched exactly once, at the very end, for the numbers reported to the client.
5. **Compared 4 model families with default-ish hyperparameters** using 5-fold stratified cross-validation on the training split only (`02_model_comparison.py`): Logistic Regression, Random Forest, LightGBM, XGBoost.
6. **Tuned all 4 families fairly** with Optuna (15 trials each, 3-fold CV on the training split, optimizing PR-AUC) so that the "winner" isn't just an artifact of untuned defaults (`03_tune_and_finalize.py`).
7. **Refit the single best family** (by tuned CV PR-AUC) on the full training split, and evaluated it **once** on the untouched test split — that's what "final metrics" means below.
8. **Explained the model's drivers** two independent ways: SHAP values and permutation importance, both computed on the held-out test set (`04_explain_drivers.py`), to make sure the "drivers" story doesn't depend on picking one particular importance metric.

All code that produced every number in this report is in this folder: `01_prepare_data.py`, `02_model_comparison.py`, `03_tune_and_finalize.py`, `04_explain_drivers.py`, run in that order.

---

## 2. What the data actually looked like (and why cleaning mattered)

This is the part of the work that mattered most. Before any modeling, EDA turned up several issues that, left unfixed, would have produced a model that looked outstanding in testing and was worthless (or actively misleading) for the retention team.

### 2.1 Two columns leak the outcome directly

- **`account_status_code`**: code "C" co-occurs with `churned=1` in 527 of 531 churned rows, and with `churned=0` in only 39 of 5,619 non-churned rows. This is effectively "were they cancelled" restated as a feature — it's set *because* the customer churned, not something known beforehand. **Dropped.**
- **`churn_reason`**: populated for churned customers only — its missing rate (91.35%) is exactly the non-churn rate. It literally records *why* a customer who churned churned (price, moved, competitor, service, other). **Dropped.**

Including either column would have produced a model with near-perfect test accuracy that is completely useless in production, since neither field exists for a customer you haven't lost yet.

### 2.2 150 exact duplicate rows

150 rows are byte-for-byte duplicates of another row (same `customer_id`, identical values everywhere) — 300 rows total. Left in place, the same customer could land in both the train and test split, silently inflating test performance (the model would partly be "tested" on a row it memorized). **Dropped (kept first occurrence), reducing 6,150 to 6,000 rows.** `customer_id` is unique after this.

### 2.3 Sentinel/placeholder values disguised as real data

- **`age`**: 33 rows have exactly 250, 29 rows have exactly 0 — both physically implausible and exactly repeated to the same value (not a natural distribution tail). Recoded to missing.
- **`total_charges`**: 367 rows (about 6%) have exactly -999.0, a classic missing-value sentinel, on top of about 4% genuine NaN. Recoded to missing.

Left as numbers, these would have distorted any model that uses `age` or `total_charges` directly (e.g., a 250-year-old customer, or -$999 in charges).

### 2.4 Inconsistent category spellings

`contract_type` had 12 raw spellings of 3 real categories (variants like "Two year " with a trailing space, "TWO YEAR" in caps, "two year" lowercase, a leading-space " One year", "Month-to-month " with trailing space, etc.). Normalized (trim + lowercase) to three clean categories: month-to-month, one year, two year. Without this, a model would treat differently-cased spellings as unrelated categories and lose most of the signal in this field — which turned out to be one of the top 3 drivers (Section 4).

### 2.5 A redundant, suspect column

`last_invoice_amount` is 97% correlated with `monthly_charges` (i.e., mostly a noisy copy of a feature already in the dataset), **and** its missingness rate is nearly double for churned customers (30.5%) vs. retained customers (16.4%) — consistent with invoices simply stopping once a customer leaves. Because *whether the field is populated at all* is entangled with the outcome, keeping it risked smuggling outcome information back in through a missingness pattern. **Dropped.**

### 2.6 Dropped as non-features

- `customer_id` — identifier, no signal.
- `signup_date` — checked whether `signup_date` plus `tenure_months` resolves to a consistent "snapshot date" across customers (which would make it a useful cohort/seasonality signal); it does not (implied snapshot dates span 2023 to 2031), indicating `signup_date` and `tenure_months` were generated independently of each other. Kept out to avoid fitting noise.

### 2.7 Final feature set (10 columns)

**Numeric:** `tenure_months`, `monthly_charges`, `total_charges`, `age`, `support_calls_90d`, `avg_daily_usage_mb`
**Categorical:** `contract_type`, `payment_method`, `region` (20 codes)
**Target:** `churned` (8.67% positive rate after cleaning — imbalanced)

Remaining missingness after cleaning: `total_charges` about 10% (genuine NaN plus recoded sentinel), `age` about 1%. Handled via median/most-frequent imputation inside the modeling pipeline (fit on train only, applied to test — no leakage).

---

## 3. Models compared, and how

**Why ranking metrics, not accuracy:** the retention team can call about 10% of customers per month, and the true churn rate is 8.7% — plain accuracy is dominated by correctly predicting "not churned" and is nearly meaningless here. This is fundamentally a **ranking/targeting problem**, so the primary metrics throughout are:
- **PR-AUC** (average precision) — robust discrimination metric under class imbalance, used to pick the winning model.
- **Precision@10%** — of the top 10% highest-risk customers the model flags, what fraction actually churn.
- **Recall@10% / capture rate** — of all customers who actually churn, what fraction are caught by calling only the top 10%.
- **Lift@10%** — Precision@10% divided by base churn rate (how much better than a random call list).
- ROC-AUC is also reported for reference but is less informative here.

**Stage 1 — baseline comparison** (`02_model_comparison.py`), 5-fold stratified CV on the training split, default-ish hyperparameters, class imbalance handled via `class_weight="balanced"` / `is_unbalance=True` / `scale_pos_weight`:

| Model | CV PR-AUC | CV ROC-AUC | CV Precision@10% | CV Recall@10% |
|---|---|---|---|---|
| Logistic Regression | 0.204 | 0.704 | 23.1% | 26.7% |
| Random Forest | 0.169 | 0.684 | 19.6% | 22.6% |
| XGBoost | 0.154 | 0.621 | 17.7% | 20.4% |
| LightGBM | 0.134 | 0.615 | 16.0% | 18.5% |

Logistic regression already led the tree ensembles at default settings — plausible on a dataset this size (4,800 training rows after the split) with a fairly weak, close-to-linear signal, where unregularized/under-tuned trees tend to overfit noise.

**Stage 2 — fair tuning of all 4 families** (`03_tune_and_finalize.py`): to make sure that wasn't just "trees need tuning," every family got an Optuna search (15 trials, 3-fold CV PR-AUC on train only):

| Model | Tuned CV PR-AUC | Best hyperparameters (abbreviated) |
|---|---|---|
| Logistic Regression | 0.2079 | C=0.024, L2 |
| LightGBM | 0.1895 | n_estimators=150, num_leaves=23, max_depth=6, lr=0.015 |
| XGBoost | 0.1850 | n_estimators=100, max_depth=3, lr=0.011 |
| Random Forest | 0.1833 | n_estimators=100, max_depth=4, min_samples_leaf=28 |

Tuning helped the tree models close some of the gap (best XGBoost depth 3, best RF depth 4 / leaf 28, best LightGBM leaves 23 — all landing on *shallow, heavily regularized* configurations, another sign the true signal is weak and close to linear), but **logistic regression remained the best model on cross-validated PR-AUC** and was selected as final.

---

## 4. Final model and how good it really is

**Final model:** `LogisticRegression(C=0.0238, penalty="l2", class_weight="balanced")` inside a pipeline that median-imputes and standardizes numeric features and most-frequent-imputes plus one-hot-encodes categoricals (all fit on train only).

**Validation scheme:** 80/20 stratified train/test split, fixed at the start (`random_state=42`); all model comparison and hyperparameter tuning used cross-validation *within the training split only*; the test split (1,200 customers, 104 of whom churned — 8.67%) was evaluated **exactly once**, after the model was fully selected and refit. These are the numbers that matter:

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.720 |
| Test PR-AUC | 0.217 (vs. 0.087 base rate) |
| Precision@10% (of the 120 customers called, how many actually churn) | 25.8% |
| Recall@10% / capture rate (of the 104 actual churners, how many are caught) | 29.8% |
| Lift@10% (vs. calling a random 10%) | 2.98x |
| Brier score (calibration) | 0.225 |

In plain terms: if the retention team calls the top 120 customers (10% of the test set) ranked by predicted risk, about **1 in 4 of those calls reaches a customer who really is going to churn** (vs. 1 in 11.5 if they called randomly), and they will have reached **roughly 3 in 10 of everyone who was actually going to churn** that month. That is genuinely useful for triage, but it is a *moderate*, not a *strong*, model — see caveats below on why, and what would likely improve it.

Supporting plots saved to `evaluation_plots.png` (ROC curve, precision-recall curve, cumulative gains/lift chart with the 10%-capacity line marked).

---

## 5. What actually drives churn

Two independent methods were used on the held-out test set so the driver story isn't an artifact of one metric:

- **SHAP values** (`feature_importance_shap.csv`, `shap_summary.png`) — magnitude *and* direction of each feature's contribution to individual predictions.
- **Permutation importance** (`feature_importance_permutation.csv`) — model-agnostic: how much test PR-AUC drops when a feature is randomly shuffled.

Both methods agree on the same top 3, in the same order:

1. **Contract type** — month-to-month customers churn far more (12.3% raw churn rate) than one-year (3.6%) or two-year (4.8%) customers. Largest permutation-importance drop (-0.072 PR-AUC) and largest SHAP contribution among the categorical features.
2. **Support calls in the last 90 days** — more support calls strongly associates with higher churn risk (churned customers average 1.63 calls vs. 1.18 for retained customers). Second-largest driver by both methods.
3. **Tenure** — newer customers churn more (churned customers average 29.7 months tenure vs. 37.3 for retained). Third-largest by both methods; the raw EDA relationship and SHAP direction agree (longer tenure leads to lower risk).

Secondary signals that showed up in SHAP but were weaker/less consistent under permutation importance (i.e., partly redundant with the top 3 once those are accounted for): **monthly charges** (higher charges associate with somewhat higher risk) and **payment method** (electronic check associates with somewhat higher risk than other payment methods). `region`, `age`, `total_charges`, and `avg_daily_usage_mb` showed negligible, close-to-noise-level importance by both methods.

**Practical read for the retention team:** the highest-leverage lever visible in this data is contract structure — month-to-month customers are the highest-risk segment by a wide margin — combined with recent support-call volume as an early-warning signal of dissatisfaction, and tenure as a baseline risk multiplier (new customers need more attention).

---

## 6. Caveats and recommendations

- **Data quality was the main risk in this task**, not model choice. Sections 2.1 to 2.5 describe issues that, if missed, would have produced a fake near-perfect model (from `account_status_code` / `churn_reason` leakage) or a quietly biased one (from duplicates, sentinels, or the `last_invoice_amount` missingness pattern). Anyone re-running or extending this pipeline should re-check these assumptions if the raw data source changes.
- **This is a moderate-strength model** (PR-AUC 0.217 vs. 0.087 base rate; lift 2.98x), not a highly discriminating one. The available features (tenure, charges, a handful of account attributes) capture some but not most of what drives an individual customer's decision to leave. Logistic regression beating tuned gradient boosting is itself informative: it suggests the *true* signal-to-noise ratio in these particular features is low and close to linear, not that a better algorithm was left undiscovered.
- **Calibration** (Brier score 0.225) was checked but not specifically optimized (e.g., via Platt scaling or isotonic regression); if the team wants the predicted probability itself (not just the ranking) to be meaningful — e.g., for expected-value-of-call prioritization — add a calibration step before deployment.
- **No time-based validation was possible**: `signup_date` did not resolve to a consistent snapshot date, so there's no reliable way to do a forward-in-time train/test split that would detect concept drift. In production, monitor live precision@10% / recall@10% monthly and retrain if they degrade — a random stratified split (as used here) can be optimistic relative to true forward-looking performance.
- **Sample size**: 6,000 customers (4,800 train / 1,200 test) is modest for 20 region categories and imbalanced classes; per-region effects in the driver analysis are noisy and shouldn't be over-interpreted individually.
- **Recommended next steps to improve the model**, in rough priority order: (1) add behavioral/trend features if available (e.g., usage trend over the last 3 to 6 months rather than a single snapshot, change in support calls month-over-month), (2) add any customer engagement or satisfaction signals (NPS, complaint text, app logins), (3) revisit `region` — 20 near-flat categories add little; consider dropping or grouping if a refresh confirms they stay uninformative, (4) once deployed, track actual call outcomes to build a proper cost/benefit-weighted threshold instead of a fixed 10%.

---

## 7. Files in this folder

- `01_prepare_data.py`, `02_model_comparison.py`, `03_tune_and_finalize.py`, `04_explain_drivers.py` — full pipeline, run in order.
- `churn_clean.csv` — cleaned dataset; `train_split.csv` / `test_split.csv` — the fixed 80/20 split used throughout.
- `cv_model_comparison.csv`, `tuning_results.json` — model comparison and tuning results.
- `final_metrics.json`, `evaluation_plots.png` — final held-out test performance.
- `feature_importance_shap.csv`, `feature_importance_permutation.csv`, `shap_summary.png` — driver analysis.
- `final_model_pipeline.joblib` — the fitted final pipeline (preprocessing + logistic regression), reusable via `joblib.load`.
