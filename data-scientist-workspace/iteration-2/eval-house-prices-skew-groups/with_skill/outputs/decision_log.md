# Decision log: House pricing model

## Stage 1 - Structure audit + stateless cleaning - GATE PASSED
Considered: exact-duplicate removal, key uniqueness check, sentinel->null, spelling normalization,
type parsing, target skew check, time coverage check (per column_audit.csv / ds-data-audit Part A).
Applied (fixed): sqft in {50,100000} -> NaN (21 rows, round implausible placeholders);
                 lot_size_sqft==9999 -> NaN (1 row); year_built in {1800,2090} -> NaN (22 rows,
                 isolated spikes >50yr from nearest real value); garage spelling variants
                 {Yes,yes,Y,YES}->Yes, {No,no,N}->No (stateless map, no info learned from data);
                 sold_date parsed to datetime; condition marked ordered categorical.
Carried forward: n/a (Stage 1 is fixed cleaning, not screened)
Skipped: dropping distance_to_center_km / tax_assessed_value / price FAR_OUT_3IQR rows as errors -
         each is a continuous distribution with no isolated spike, consistent with real extreme
         properties (expensive homes, far-flung addresses); deferred to Stage 5 as extreme-not-error.
Evidence: code/01_stage1_audit.py, column_audit.csv, cleaning_log.md
Open risks: property_id repeats 320x (7.4% of rows) -> must group-split at Stage 2 or tax_assessed_value/
            lot_size_sqft (static per property) will leak across train/test for the same property.
            price_per_sqft flagged DERIVED_FROM_TARGET here on formula grounds; formal leak-table
            confirmation deferred to Stage 3 as the skill requires (kept in data through Stage 2, never
            used as a feature).
## Stage 2 - Validation design - GATE PASSED
Considered: stratified, random, group, time, group_time (per ds-validation-design deployment table).
Deployment answer: new listings are mostly entirely-new properties, but 320/4320 (7.4%) rows are
resales of a property already in the data, and prices plausibly drift over 4 years (2021-2024) ->
"entities repeat and time matters" -> group_time.
Applied (fixed): group_time split, group=property_id, time=sold_date, test=most recent 15% by date
                 (2024-05-11..2024-12-30, 648 rows), holdout=next-most-recent 15%
                 (2023-10-18..2024-05-10, 649 rows), train=earliest ~70% (2021-01-01..2023-10-17).
                 5-fold expanding-window CV on train (folds: 1438/312 .. 2645/306).
                 Group-overlap fix: 72 train rows whose property_id also appears in holdout/test
                 were EXCLUDED from train (conservative "new entity" deployment policy) rather than
                 left in, since tax_assessed_value/lot_size_sqft/sqft are static per property and
                 would otherwise let the model memorize the exact holdout/test row's attributes
                 from its own earlier sale. holdout&test overlap = 0 (clean).
Carried forward: n/a (split is fixed infrastructure, not screened)
Skipped: pure `time` scheme (ignores property_id repeats - rejected, would leave the leak above in
         place); pure `group` scheme (ignores 4-year drift and would validate on a time-shuffled
         mix, understating drift risk for a "future listings" tool); `stratified`/`random`
         (rows are not independent - 7.4% repeat properties - and deployment is on a future period).
Noise floor: Ridge/log-price baseline, deterministic given fixed expanding folds -> noise_floor_sd
             = fold_sd/sqrt(5) = $289.57 (fold_sd $647.49, mean CV MAE $13,612). Differences between
             later pipelines smaller than ~$290 (on shared folds, paired SE) are treated as ties.
Evidence: code/02_stage2_validation.py, split/split.csv, split/split_spec.json, split/noise_floor.json
Open risks: test set is a single 8-month chronological slice (648 rows); if the market moved
            unusually in H2 2024 the test score reflects that period specifically, not a
            representative average year - flagged for Stage 9 slice analysis by sold_date quarter.
## Stage 3 - Leakage hunt + relationship EDA (train only) - GATE PASSED
Considered: univariate leak screen (leak_screen.py, GroupKFold by property_id), name-pattern check
for after-the-fact columns, adversarial validation (train vs test), redundancy/VIF-style checks,
stability across the train/test time boundary.
Applied (fixed): DROP price_per_sqft - DERIVED_FROM_TARGET? pair with sqft (R2=1.000 given price);
                 confirmed arithmetically (price_per_sqft == price/sqft up to rounding) and by
                 definition only computable after the sale closes -> not available at listing time.
                 All other 10 candidate features kept as legit (property-record attributes,
                 available before/at listing; see leak_table.csv `recorded_when`/`evidence`).
Carried forward: n/a (leak removal is fixed, never searched)
Skipped: treating tax_assessed_value as suspect despite its top score (0.937) - it is a public
         pre-sale tax record, not target-derived; kept as the strongest legitimate predictor.
Evidence: code/leak screen output, leak_table.csv, eda_notes.md
Adversarial validation: train(2021-2023) vs test(2024) AUC = 0.608 via HistGradientBoostingClassifier,
         5-fold CV; expected for a time split (measures real market drift, not leakage) - no single
         feature scores above 0.53 individually, so the drift is diffuse, not one bad column.
         Recommend adding sold_year/sold_month features so the model can represent the trend.
Open risks: `condition` timing is ASSUMED to be set at listing (agent/inspector assessment), not
            confirmed with a data owner; if it is instead assigned post-sale it would need review.
## Stage 4 - Missing values - GATE PASSED
Diagnosis (missing_profile.csv, train n=2951): lot_size_sqft 20.6% missing, garage 1.4%, sqft 0.5%,
year_built 0.5% - all MCAR-like (not predictable from other columns, AUC/R2 near chance;
target_signal ~0.00-0.02; no missing-together pattern; no time trend). No informative or
structural missingness found.
Considered: drop rows, median, median+indicator, KNN, iterative(MICE), native NaN (GBDT) - per
ds-missing-values Step 2 table for MCAR-like data.
Applied (fixed): n/a (imputers are searched, not fixed)
Carried forward: median (linear, MAE $13,612 +/- $647); median (gbdt, MAE $12,471 +/- $354)
Skipped: median+indicator (linear) - tied with median (diff -$28 vs corrected SE $36); simpler
         option wins per the ties-go-to-simpler rule, consistent with zero measured target signal
         from missingness; native_nan (gbdt) - measured worse by $93 MAE, 1.4 corrected SE, a real
         (if small) gap, not a tie; KNN and iterative(MICE) - not screened (BUDGET CUT, logged):
         diagnosis shows no MAR structure (other columns do not predict missingness, max
         predictable-from-obs AUC 0.58) so there is little for a model-based imputer to borrow from,
         and the skill explicitly says simple imputers usually match MICE for boosted trees; time
         did not allow screening them to confirm, so this is a logged gap, not a hidden one;
         drop rows - 22.6% of train rows have at least one missing feature, far above "few rows",
         and the pipeline must handle missingness at inference regardless.
Evidence: code/03_stage4_missing.py, missing_profile.csv, stage4_missing_screen_linear.csv,
          stage4_missing_screen_gbdt.csv, search_space.json (stages.missing_values)
Open risks: KNN/iterative imputers unscreened due to time budget; recommend screening them if the
            final model needs to be pushed past this pass (see report Next steps).
## Stage 5 - Outliers - GATE PASSED
Verdicts (outlier_profile.csv, train n=2951): sqft and tax_assessed_value extremes are SIGNAL
(lift 2.3x, 2.6x vs overall mean target) -> keep untouched, both families. lot_size_sqft and
distance_to_center_km are heavy tails (skew 1.79, no domain-impossible values) -> transform/winsorize
candidates for the linear proxy. price (target) has extremes (lift 2.8x) -> already handled by
modeling log(price) (Stage 0/6 decision), never dropped from any split. No column showed a 10x/100x
isolated spike here (those were already caught and nulled as sentinels at Stage 1).
Considered: none, Winsorizer(0.01,0.99), RobustClipper, log/Yeo-Johnson on tailed columns,
AnomalyScore feature (per ds-outliers Step 3 table).
Applied (fixed): none (thresholds are screened, not fixed, except the already-decided log target)
Carried forward: winsorize_p01_p99 on {lot_size_sqft, distance_to_center_km} (linear, MAE
                 $13,543 +/- $578); none (linear, MAE $13,612 +/- $647, kept alongside as provisional
                 - only 5 folds to pair on, effect within the global noise floor even though the
                 paired-fold SE marks winsorize ahead); none (gbdt, fixed reference, MAE $12,471 +/-
                 $354 - not screened, see below)
Skipped: RobustClipper / log-transform of tail columns - not screened (BUDGET CUT): winsorize
         already covers "limit leverage of the tail" for the linear proxy and the two are unlikely
         to differ beyond this data's noise floor; AnomalyScore feature - deferred to Stage 6 as a
         feature-engineering candidate rather than an outlier treatment; treatments on the GBDT
         proxy - not screened because tree splits are invariant to monotonic transforms/clipping of
         feature values (ds-outliers: "Splines depend on order, not distance"), so a screen would be
         expected to return a null result at the cost of the remaining budget.
Evidence: code/04_stage5_outliers.py, outlier_profile.csv, stage5_outlier_screen_linear.csv,
          search_space.json (stages.outliers)
Open risks: sqft/tax_assessed_value extremes kept as signal on outlier_profile.csv's lift evidence;
            worth a manual look at the top 1% IsolationForest rows (mean target $248,897 vs $109,107
            overall) before shipping, to rule out a handful of luxury-property data entry errors.
## Stage 6 - Feature engineering - GATE PASSED
Considered: domain ratios/derived (house_age, total_rooms), calendar parts (sold_year, sold_month),
log/Yeo-Johnson transforms of skewed numerics (linear only, trees invariant to them - not screened,
same mechanics reason as Stage 5), neighborhood encoding (one-hot vs cross-fitted TargetEncoder),
automated feature generation (not attempted - BUDGET CUT, only 9 raw predictors so hand-built
features already cover the obvious ratios/interactions a generator would find).
Applied (fixed): n/a (all screened)
Carried forward: plus_engineered={house_age, total_rooms, sold_year, sold_month} + base columns
                 (linear, MAE $12,594 +/- $605, a real $949 gain over base_cols, 1 SE test passes
                 clearly - base_cols dropped for linear); plus_engineered (gbdt, MAE $12,397 +/- $338,
                 tied with base_cols $12,408 +/- $368 - kept anyway over the tied base_cols because
                 Stage 3's adversarial-validation AUC 0.608 shows real train/test drift across
                 sold_date, and sold_year/month is the only way the model can represent that trend
                 at deployment on future listings, even though it does not help on this particular
                 historical CV split); onehot encoding for neighborhood (linear, MAE $12,594 +/- $605)
Skipped: base_cols (linear) - real loss vs plus_engineered (diff $949 >> corrected SE, not a tie);
         target_encoding for neighborhood (linear) - real loss vs one-hot (MAE $13,169 vs $12,594,
         diff beyond both the paired SE and the global noise floor); likely cause: 30 levels over
         2951 rows means several neighborhoods have well under 50 training rows, and cross-fitted
         encoding's per-fold estimate is noisier there than a plain one-hot column that Ridge's L2
         penalty already regularizes; log/Yeo-Johnson transforms of sqft/lot_size/tax_assessed -
         not screened (BUDGET CUT): winsorizing the same tail columns at Stage 5 already captured
         most of the "limit leverage of the tail" benefit, and stacking both was judged unlikely to
         clear the noise floor given the time remaining.
Evidence: code/05_stage6_features.py, feature_ledger.csv, stage6_feature_ablation_linear.csv,
          stage6_feature_ablation_gbdt.csv, stage6_encoding_screen_linear.csv,
          search_space.json (stages.features, stages.encoding)
Open risks: condition's availability-at-listing is still ASSUMED (see Stage 3); if it is actually
            recorded post-sale it should be dropped and the model rescreened.
## Stage 7 - Baselines + model families + imbalance - GATE PASSED
Data size 4320 rows -> ds-model-selection "up to ~5,000 rows" band: recommends TabPFN-3.5/TabICLv2,
CatBoost, regularised linear. Not a classification problem, so imbalance handling (Step 4) does not
apply. Target transform: log(price) via TransformedTargetRegressor (Step 5, "skewed positive"
candidate), already validated at Stages 2-6.
Considered (baselines): DummyRegressor(median), business rule ($/sqft by neighborhood x sqft),
Ridge, DecisionTreeRegressor(max_depth=3). Considered (families): regularised linear (Ridge),
shallow tree, LightGBM, CatBoost, TabPFN-3.5/TabICLv2 foundation models.
Applied (fixed): n/a
Carried forward: lightgbm_final_prep (MAE $12,397 +/- $338, best); ridge_final_prep (MAE $12,594 +/-
                 $605, tied with LightGBM within the corrected paired SE) - both to Stage 8.
Skipped: tree_depth3 - MAE $15,641, clearly worse than Ridge/LightGBM beyond noise, kept only as a
         baseline reference, not carried to Stage 8; CatBoost - attempted and dropped: fitting
         CatBoostRegressor(cat_features=[...]) inside screen.py's evaluate() raises
         "RuntimeError: Cannot clone object ... constructor either does not set or modifies
         parameter cat_features" from sklearn.base.clone() - a confirmed library incompatibility
         (checked by running it, per the skill's own "library facts are checked, not recalled"
         rule), not assumed; would need a clone-safe wrapper or manual per-fold fit/copy to use,
         which the remaining budget did not allow - logged as a next-step recommendation since
         CatBoost typically outranks LightGBM on TabArena; TabPFN-3.5/TabICLv2 - recommended by
         data size but NOT INSTALLED and installing is not permitted in this environment; also
         TabPFN-2.5/3.5 weights are non-commercial-licensed (checked via the skill's reference),
         which would block use in a commercial pricing tool regardless - both logged as skipped
         with the install/licence note, listed in report Next steps.
Learnability verdict: LEARNABLE. Best families (LightGBM $12,397, Ridge $12,594) beat the business
         rule ($17,507) by ~$5,000, far beyond the $290 noise floor, and beat the trivial baseline
         by >3x. Learning curve (25/50/75/100% of train) is still gently falling at full size for
         both families (LightGBM 13,054->12,626; Ridge 12,724->12,610 from 75%->100%) -> mildly
         data-limited, not flat: more historical sales would likely help further, but the model is
         clearly learning real structure now.
Evidence: code/06_stage7_models.py, stage7_baseline_family_screen.csv, search_space.json (stages.models)
Open risks: only 5 CV folds/pairs throughout Stages 4-7 (screen.py's own low-power warning); Ridge
            and LightGBM are statistically tied, so Stage 8's holdout check is the tie-breaker.
## Stage 8 - Joint pipeline search - GATE PASSED (custom narrow search, budget-adapted)
Search space (collapsed by Stages 4-6 to 2 free slots): model family {ridge, lightgbm} x
ridge{alpha log-uniform[0.01,100], outlier treatment [none, winsorize]} x lightgbm{n_estimators
[100,400], num_leaves[7,63], learning_rate log[0.01,0.2], min_child_samples[5,50]}. missing-value
strategy fixed to median (Stage 4 sole survivor both families); features fixed to plus_engineered;
encoding fixed to one-hot. 25-trial Optuna TPE study (BUDGET CUT vs the skill's 40-60 trial
guideline: the space was already narrow after Stages 4-6, and 1 seed/trial was used instead of 3 -
Ridge is deterministic and LightGBM was seeded at a fixed random_state to fit the time remaining).
25 trials completed in 25.8s (0.1-2.6s/trial).
Best-by-mean: LightGBM (n_estimators=250, num_leaves=21, lr=0.121, min_child=43), CV MAE $12,108 +/-
              $593 (5 folds).
Runner-up (Ridge best trial): alpha=0.0115 (near-OLS), outlier=winsorize, CV MAE $12,562 +/- $597.
1-SE rule: paired per-fold diff (ridge - lightgbm) = $454, corrected SE = $488 -> TIE (diff < SE) ->
           tie goes to the simpler, cheaper, more interpretable option = RIDGE. PICK = Ridge
           (alpha=0.0115, winsorize).
Locked-holdout check (649 rows, 2023-10-18..2024-05-10, never used before this check):
  pick (Ridge):     CV MAE $12,562 -> HOLDOUT MAE $12,331 +/- $454  (gap $231, well within 2 SE -> OK)
  best_by_mean (LightGBM): CV MAE $12,108 -> HOLDOUT MAE $13,003 +/- $521 (gap $895, larger than its
                            CV spread - the CV advantage did NOT transfer to the holdout period)
The holdout confirms the 1-SE-rule pick: Ridge generalizes at least as well as LightGBM on unseen,
later-period rows, while LightGBM's apparent CV edge looks like mild overfitting to the 5 CV folds
(consistent with LightGBM's higher hyperparameter count on a mid-size, mostly-linear-signal dataset -
exactly the ds-model-selection red flag "on small or mostly linear signals a regularised linear
model often wins").
FINAL PIPELINE: log-target Ridge(alpha=0.0115) on {sqft, bedrooms, bathrooms, year_built,
lot_size_sqft, distance_to_center_km, tax_assessed_value, house_age, total_rooms, sold_year,
sold_month} (median-imputed, lot_size_sqft/distance_to_center_km winsorized at 1%/99% in-fold,
standardized) + one-hot {neighborhood, condition, garage} (most-frequent-imputed).
Evidence: code/07_stage8_joint_search.py, search/trials.csv, search/finalists.csv, search/result.json
Open risks: only 25 trials and 1 seed - a larger search (60+ trials, 3 seeds) might still favor
            LightGBM or find a better Ridge alpha; recommended as a next step if more time is available.
## Stage 9 - Evaluation + sign-off - GATE PASSED
Preconditions: verify_test_untouched("split")==True (checked in code). Point-prediction model
(Ridge, alpha=0.0115, winsorize) fit on TRAIN ONLY (2951 rows) - a deliberate deviation from "refit
on train+holdout": the holdout must stay unseen by this exact fitted model to serve honestly as the
split-conformal calibration set (see script docstring). No threshold/calibrator step (regression,
not classification).
Test metrics (648 rows, 2024-05-11..2024-12-30), bootstrap 95% CI, vs baselines on the SAME rows:
  trivial_median:   MAE $42,385 [39,157-45,654]  RMSE $60,414  MAPE 37.2%
  business_rule:    MAE $20,158 [18,650-21,661]  RMSE $27,994  MAPE 16.5%
  FINAL (Ridge):    MAE $13,650 [12,490-14,918]  RMSE $20,719  MAPE 12.1%
Test MAE ($13,650) sits a bit above the Stage 8 CV mean ($12,562) and holdout ($12,331) but their
95% CI overlaps both; not "too good to be true" (it is worse than CV, if anything), and it clears
the business-rule baseline by ~$6,500, far beyond the $290 noise floor -> SUCCESS THRESHOLD MET
(beats baseline beyond noise; see open risk on coverage below for the interval half of the threshold).
Uncertainty (Step 4): split-conformal, hand-implemented (MAPIE not installed - environment
constraint). Nonconformity score = |log(y) - log(pred)| on the HOLDOUT (n=649); log-space chosen
because dollar-scale residuals are heteroscedastic (bigger absolute errors for bigger houses) and
log residuals are close to homoscedastic, giving naturally wider dollar intervals for expensive
homes without extra machinery. Finite-sample-corrected quantile at level ceil(650*0.90)/649 = 0.9014,
q=0.2295. TEST interval coverage = 88.7% (target 90%, ASSUMED acceptance band [87%,93%] from
problem_statement.md) -> WITHIN BAND, slightly under nominal; asymmetric miss (8.0% of actual prices
fell below the lower bound vs 3.2% above the upper bound) - the interval is somewhat too tight on
the downside. Mean interval width $54,714 (about half the median sale price) - wide but usable as a
quoting range, not a precise number.
Slices (Step 6): worst by MAE - top price quintile ($26,749, expected: absolute error scales with
price level), neighborhoods N02/N03/N07/N09/N10 ($17-22k, smaller sample neighborhoods), condition
Poor ($16,460, n=39, likely the hardest-to-price segment). No slice catastrophically worse than the
others relative to its price level; no fairness/protected-attribute columns present in this data.
Error analysis (Step 7): worst_errors.csv holds the 20 largest-residual test rows for manual review
(not reviewed row-by-row here - BUDGET CUT, logged as a next step).
Sanity tests (Step 8), all PASS: shuffled-label CV MAE collapses to $38,586, matching the trivial
baseline ($38,270) - confirms no row-order/ID leak; duplicate row -> identical predictions; 5
deliberately-uncleaned raw rows (spelling variant, sentinel sqft, missing lot_size) score without
error through clean_raw()+final pipeline; increasing sqft by 500 raises the prediction ($312,763 ->
$363,789) - directionally sane.
Explanations (Step 9): permutation importance on the holdout - neighborhood ($20,068 MAE increase)
and sqft ($18,620) dominate; tax_assessed_value ranks last ($194) despite its high Stage-3
univariate score (0.937) because it is collinear with sqft/neighborhood and a linear model splits
credit among correlated features (flagged explicitly, matches ds-evaluation's own warning) - not a
leak concern, tax_assessed_value's information is simply redundant with two stronger features here.
Evidence: code/08_stage9_evaluation.py, test_metrics.json, slice_table.csv, worst_errors.csv,
          permutation_importance.csv
Open risks: interval coverage (88.7%) is slightly under the 90% nominal target and its miss is
            asymmetric (more low-side misses); a next iteration could widen the conformal margin
            (target 92-93% nominal to land near 90% empirical) or use conformalized quantile
            regression for a data-driven asymmetric interval instead of the log-space approximation
            used here. Error-analysis rows were exported but not manually reviewed (budget).
## Stage 10 - Handoff - GATE PASSED
Production pipeline (code/production_pipeline.py): raw CSV row -> clean_raw (Stage 1 stateless
rules) -> calendar parts -> AddDomainFeatures -> winsorize/median-impute/standardize numerics,
most-frequent-impute+one-hot categoricals (all fitted in-fold) -> Ridge(alpha=0.011545) on
log(price). Fitted on RAW train-part rows only (2951 rows; NOT train+holdout - see Stage 9 design
note: the holdout must stay unseen by the shipped model so the conformal margin it calibrated
remains valid). Saved with skops (`production_pipeline.skops`); pickle/joblib avoided per the
skill's own security note (arbitrary code execution on load from untrusted sources). Reloaded
successfully; untrusted-type list is limited to this project's own classes/functions plus
numpy.dtype (expected, trusted since we authored them).
Inference-readiness (all PASS): 10 raw, uncleaned rows (unmodified from the source CSV, including
whatever spelling variants/sentinels naturally occur) scored end to end at ~2.1 ms/row; a row fed
twice gives bit-identical predictions; a row with every feature column blanked still returns a
(fallback-median-driven) prediction rather than erroring.
scikit-learn 2.3 / lightgbm 4.7 / xgboost 3.4 pinned via the environment description; seeds fixed
(random_state=0 throughout screening, 42 for the split and the Optuna sampler).
Evidence: code/09_stage10_handoff.py, code/production_pipeline.py, production_pipeline.skops
Open risks: shipped model uses only 2951 of the 4320 rows (holdout+test withheld for calibration/
            evaluation); once this model is retired for a future retrain, the data owner can retrain
            on all 4320+ rows with a fresh train/calibration/test split for a small further accuracy
            gain, since the current test and holdout sets are now "spent" for model selection.
