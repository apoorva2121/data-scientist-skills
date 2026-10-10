# Decision log

## Stage 0 - Frame the problem - GATE PASSED
Considered: rule-based heuristic vs ML model, metric choice (accuracy vs precision@k vs PR-AUC vs ROC-AUC).
Applied (fixed): target = `churned`; primary metric = precision@10% (capacity constraint from user);
                 secondary = PR-AUC, ROC-AUC, recall@10%, calibration.
Carried forward: n/a (framing stage)
Skipped: accuracy as primary - meaningless under 8.65% base rate and irrelevant to a capacity-constrained
         call list (rejected in favor of precision@10%).
Evidence: problem_statement.md
Open risks: 5 ASSUMED lines (see problem_statement.md); user unavailable to confirm this run.
Time budget: 25 min total. Stage 0-3 budget ~6 min, Stage 4-7 ~9 min, Stage 8 ~6 min, Stage 9-10 ~4 min.
Given the severity of this budget vs a normal project, breadth is cut throughout: CLI helper scripts
are replaced with compact equivalent code in fewer files, CV uses 3 folds x 2 repeats (not 5x5), and
the Stage 8 search is a small manual grid (not Optuna). Correctness gates (leak table, split
discipline, test-once) are never cut.

## Stage 1 - Structure audit + stateless cleaning - GATE PASSED
Considered: sentinel detection, spelling-variant canonicalization, duplicate handling, impossible-value nulling.
Applied (fixed): total_charges -999->NaN; age 0/>=120->NaN; contract_type case/whitespace canonicalized
                 to 3 levels; 150 exact full-row duplicate rows dropped (6150->6000).
Carried forward: n/a (stateless only, no fitted statistic)
Skipped: imputation - explicitly deferred to Stage 4 (must be fit in-fold, not here).
Evidence: code/01_03_audit_split_leak.py, column_audit.csv, cleaning_log.md
Open risks: duplicate rows assumed to be double-counted extracts of the same event (all columns
identical); churn rate stable pre/post dedup (8.65% -> 8.67%).

## Stage 2 - Validation design + carve test - GATE PASSED
Considered: stratified, grouped, time-based, group_time schemes.
Applied (fixed): stratified 65/15/20 train/holdout/test split (3900/900/1200 rows), seed=42, frozen
                 test IDs hashed (split_spec.json). Noise floor measured on a simple legit-feature
                 logistic pipeline: RepeatedStratifiedKFold(3,3) ROC-AUC mean 0.6895, fold_sd 0.0185,
                 repeat_sd (noise_floor_sd) 0.0024.
Carried forward: n/a
Skipped: group split - customer_id unique per row after Stage 1 dedup, no entity repeats to group on.
         time split - no explicit future-period deployment stated and no drift check run given the
         time budget (ASSUMED cross-sectional snapshot); logged as an open risk, not verified.
Evidence: code/01_03_audit_split_leak.py, split.csv, split_spec.json, search_space.json
Open risks: time-based drift was not checked (budget cut); if churn definition changed over the
2023-2025 signup window this split could be optimistic. Adversarial validation train-vs-test
AUC = 0.5014, consistent with a sound iid split.

## Stage 3 - Leakage hunt + relationship EDA (train only) - GATE PASSED
Considered: univariate AUC screen, near-perfect/dominant flags, informative-missingness, record-timing
            evidence, adversarial validation.
Applied (fixed): dropped churn_reason (univariate AUC 1.000, non-null iff churned=1 - written after
                 the outcome); dropped account_status_code (AUC 0.995 for status=='C', a cancellation
                 status recorded after the outcome).
Carried forward: legit numeric/categorical features (support_calls_90d AUC 0.592, monthly_charges
                 0.572, last_invoice_amount 0.568, avg_daily_usage_mb 0.503, age 0.490, total_charges
                 0.434, tenure_months 0.389) - all univariate, none near-perfect, no leak flags.
Skipped: last_invoice_amount drop - considered but kept ASSUMED legit (billed amount known at scoring
         time); flagged as the top open risk for the data owner to confirm, since if it is instead a
         final closing invoice it would be a leak like account_status_code.
Evidence: code/01_03_audit_split_leak.py, leak_table.csv, eda_notes.md
Open risks: last_invoice_amount timing unconfirmed (see problem_statement.md open risk #2).

## Stage 4 - Missing values - GATE PASSED
Considered: drop rows, median, median+indicator, KNN, iterative(MICE), native NaN (GBDT).
Applied (fixed): none new (sentinel->NaN was done stateless at Stage 1).
Carried forward: linear: median (ROC-AUC 0.6827+/-0.0155) and median+indicator (0.6801+/-0.0174) -
                 tied within noise (diff 0.0026 ~ noise_floor_sd 0.0024), both kept; gbdt: native_nan
                 (0.6142+/-0.0111) and median+indicator (0.6164+/-0.0145) - tied, native_nan preferred
                 as simpler/cheaper (no fitted imputer needed).
Skipped: KNN/iterative(MICE) - columns are weakly inter-correlated (predicting-other-cols AUC ~0.50
         for total_charges/age), so MAR-style borrowing has little to gain; dropped given the time
         budget rather than screened. drop rows - would still need to handle NaN at inference
         (violates rule 2 in ds-missing-values).
Evidence: code/04_06_missing_outliers_features.py, missing_profile.csv, search_space.json
Open risks: last_invoice_amount indicator diagnosed "informative" (target AUC 0.561 vs 0.5), but its
inclusion did not clearly beat "no indicator" in the aggregate CV screen at this sample size (~3900
train rows) - kept the indicator anyway per the mechanism diagnosis, per skill guidance
("informative missingness is signal").

## Stage 5 - Outliers - GATE PASSED
Considered: none, Winsorizer, RobustClipper, log1p/Yeo-Johnson, AnomalyScore.
Applied (fixed): none as row/cell removal - no column violates a domain rule beyond what Stage 1
                 already nulled (age, total_charges sentinels).
Carried forward: avg_daily_usage_mb: "none" and "log1p" tied for linear (0.6801 vs 0.6830, diff
                 0.0029, at the noise floor) - both kept; not screened for gbdt (trees invariant to
                 monotonic transforms, per skill red flag).
Skipped: Winsorizer/RobustClipper - avg_daily_usage_mb's extreme tail carries a lift of 1.17x
         (extremes churn slightly more, not a data-entry error signature), monthly_charges/
         total_charges/support_calls_90d/last_invoice_amount extremes carry lift 1.6-2.9x (signal,
         not error) - capping would remove real signal, so not screened.
Evidence: code/04_06_missing_outliers_features.py, outlier_profile.csv, search_space.json
Open risks: none material.

## Stage 6 - Feature engineering - GATE PASSED
Considered: ratio/domain features (charge_gap, is_new_customer), categorical encoding (one-hot for
            all 3 low/mid-cardinality categoricals), selection (VarianceThreshold - skipped, no
            near-constant columns).
Applied (fixed): one-hot encoding for contract_type/payment_method/region (fitted in-fold via the
                 pipeline's ColumnTransformer).
Carried forward: base_plus_engineered (charge_gap + is_new_customer) clearly helps gbdt (0.6282 vs
                 0.6142 native, +0.014, well above noise_floor_sd 0.0024); ties for linear (0.6825 vs
                 0.6801, diff 0.0024 = noise floor). Carried forward as the primary feature set for
                 both families given the clear gbdt gain and no linear cost.
Skipped: target encoding for region - only 20 levels, one-hot at n=3900 leaves ~195 rows/level, no
         sparsity problem that would justify target encoding's extra leak surface. Interaction terms/
         splines - cut for time budget (would need per-pair screening); flagged as a next step.
Evidence: code/04_06_missing_outliers_features.py, feature_ledger.csv, search_space.json
Open risks: GBDT (HistGradientBoostingClassifier, default-ish settings) scores meaningfully below
logistic regression on every candidate so far (~0.61-0.63 vs ~0.68 ROC-AUC, gap >> noise floor) -
consistent with the skill's "small or mostly linear signals" pattern; Stage 7-8 must not assume GBDT
wins by default and must tune it properly before ruling it out.

## Stage 7 - Baselines + model families + imbalance - GATE PASSED
Considered baselines: trivial (DummyClassifier prior), business rule (support_calls_90d + month-to-
            month flag), simple model (logistic, balanced), shallow tree (depth=3).
Considered families: regularised logistic regression, LightGBM, XGBoost, shallow decision tree.
                      Skipped GBDT variants beyond LightGBM/XGBoost (CatBoost, HistGBM) for time; kept
                      the two most standard boosted implementations.
Applied (fixed) imbalance handling: class_weight="balanced" (logistic, tree, LightGBM) /
            scale_pos_weight=neg/pos (XGBoost) - step 1 of the imbalance ladder (metric+threshold
            change) plus class weights (step 2); resampling (SMOTE) not screened - published evidence
            cited in the skill (resampling rarely beats threshold tuning for strong classifiers) and
            time budget.
Baseline table (precision@10%, primary metric): trivial 0.087, business rule 0.164, simple logistic
            0.208, shallow tree 0.185.
Family screen (precision@10%): logistic 0.208 >> LightGBM 0.138 > XGBoost 0.121 (defaults). Logistic
            beats every baseline and every boosted-tree default by more than the noise floor
            (noise_floor_sd=0.0024 on ROC-AUC scale; the gap here is 0.06-0.09 on precision@10%, far
            larger). Learning curve (logistic, ROC-AUC): 0.593 -> 0.650 -> 0.668 -> 0.680 across train
            sizes 650->2600, still rising at full size -> data-limited, more rows would likely help
            (recorded in report next steps).
Skipped: TabPFN/foundation models - CPU-only fits the <5,000-row guidance, but (a) TabPFN weights are
            non-commercial and this is a client project (licence not cleared), (b) hosted inference
            would send customer rows to a third party and the user was unavailable to approve that
            data-location choice. Logged as a next-step recommendation, not silently dropped.
            Resampling (SMOTE/undersampling) - not screened, see imbalance note above.
Evidence: code/07_08_models_search.py, baseline_table.csv, family_screen.csv
Open risks: GBDT families underperform logistic by a wide, consistent margin across every screen so
far (Stages 4-7) - unusual but consistent with a weak/mostly-linear signal at this row count; flagged,
not just accepted, and re-checked at Stage 8 with a small hyperparameter grid before ruling GBDT out.

## Stage 8 - Joint pipeline search - GATE PASSED
Considered: small manual grid (not Optuna, given the time budget) crossing model family x key
            hyperparameters x the Stage 4-6 carried-forward preprocessing (median+indicator impute,
            log1p usage for linear, engineered features): 3 LightGBM configs, 2 XGBoost configs, 3
            logistic C values = 8 candidates, 3-fold CV, precision@10% as primary (paired per fold).
Applied (fixed): same fixed cleaning/leak-drop/split as Stages 1-3.
Carried forward / winner: logreg_C0.01 (precision@10% CV 0.2333 +/- 0.0617 over 3 folds; ROC-AUC
            0.6871). Runner-up logreg_C1.0 (0.2154 +/- 0.0335; ROC-AUC 0.6790) - within noise of the
            winner (both logistic; the C sweep is essentially a tie, C=0.01 picked as nominal best and
            as the more-regularised, simpler option). Best boosted-tree candidate (xgboost_d3_lr0.05)
            reached only 0.1923 - below the logistic runner-up.
Locked-holdout check (900 rows, never used for selection until now): logreg_C0.01 precision@10%=0.200,
            ROC-AUC=0.7433; logreg_C1.0 precision@10%=0.167, ROC-AUC=0.7288. Winner ranks above
            runner-up on holdout in both metrics - consistent with the CV ranking, no overfitting flag.
Skipped: wider hyperparameter search (Optuna study) - time budget; the small grid already shows a
            clear, stable family-level winner (logistic) with tree-based options far behind, so
            further tuning of the losing families was not worth the remaining budget.
Evidence: code/07_08_models_search.py, search_log.csv, search_space.json (winner/runner_up fields)
Open risks: precision@10% on 3 folds of ~1300 rows (~130 calls, ~11 expected positives/fold) is a
noisy statistic (sd 0.03-0.07) - the CV ranking among logistic C values should not be over-trusted;
the Stage 9 test-set CI is the number to report, not this CV table.

## Stage 9 - Evaluation + sign-off - GATE PASSED
Considered: threshold at action-capacity (top 10%, chosen on holdout scores, value 0.6656), no
            probability calibration performed given weighting distortion is expected (class_weight=
            "balanced") but probabilities are consumed only for ranking (top-k), not as calibrated
            probabilities, so calibration is reported (Brier 0.223, reliability table) but not fixed.
Applied (fixed): refit winner (logreg, C=0.01, class_weight="balanced", median+indicator impute,
            engineered features, one-hot categoricals) on train+holdout; test set scored exactly once
            after verify_test_untouched (sha256 match confirmed).
Test results (95% bootstrap CI, n=1200, 104 positives): precision@10% = 0.267 [0.192, 0.350] (called
            n=120), recall captured@10% = 0.308, ROC-AUC = 0.718 [0.667, 0.768], PR-AUC = 0.213
            [0.161, 0.301], Brier = 0.223, lift over base rate (8.67%) = 3.08x. Business rule baseline
            on the same test rows: precision@10% = (see baseline_table.csv train estimate 0.164;
            test-rows rule score computed in code/09_10, consistent direction). Meets the ASSUMED
            success threshold (>=3x base rate) marginally (3.08x).
Agreement check: CV 0.233 -> holdout 0.200 -> test 0.267, all within each other's noise given the
            small per-fold k (~11-13 expected positives among ~120-130 calls); no "too good to be
            true" pattern, ROC-AUC progression (0.687 CV / 0.743 holdout / 0.718 test) also consistent
            - no leak resurfaced.
Slices: worst by discrimination = One-year-contract customers (ROC-AUC 0.528, near chance; precision
            @10% 0.071 vs slice base rate 0.028, still a 2.5x lift but the model is weak here).
            Best = 0-12 month tenure customers (precision@10% 0.364, 4.2x their own base rate).
Error analysis: false positives are short-tenure, month-to-month, high-support-call customers who did
            not churn this period (a plausible risk profile, not a data error); false negatives are
            long-tenure One/Two-year customers with no support-call activity who nonetheless churned -
            the available features carry no warning signal for this group (a feature gap, not a
            modelling gap) - recommended as the top next step.
Sanity tests: shuffled-label PR-AUC 0.096 (~= base rate 0.087, passes); duplicate-row identical
            prediction (passes); 3 raw uncleaned rows scored end-to-end without error (passes);
            repeat scoring of one row identical (passes).
Explanations: standardized logistic coefficients and permutation importance (roc_auc, test set, 20
            repeats) agree on the top drivers: contract_type (esp. Month-to-month, positive), tenure
            (negative/protective), support_calls_90d (positive). None of the top features are leak
            suspects from the Stage 3 leak table.
Evidence: code/09_10_evaluation_handoff.py, test_results.json, calibration_table.csv, slice_table.csv,
            sanity_checks.json, coefficients.csv, permutation_importance.csv
Open risks: precision@10% test CI is wide (0.19-0.35) at n=1200/104 positives - a larger test set
would narrow it; One-year-contract churn is essentially unmodelled with current features.

## Stage 10 - Handoff - GATE PASSED
Applied: full pipeline (stateless clean_raw equivalent logic + median/indicator imputer + one-hot +
            logistic C=0.01) saved to final_model.joblib with the capacity threshold (0.6656) and
            library versions pinned in the file; 3 raw rows and 1 duplicate row scored correctly
            end-to-end (Stage 9 sanity tests double as the Stage 10 inference-readiness check).
Evidence: code/09_10_evaluation_handoff.py, final_model.joblib, model_card.md, report.md
Open risks: see problem_statement.md ASSUMED lines - target window, last_invoice_amount timing, and
success threshold all need data-owner confirmation before this ships.
