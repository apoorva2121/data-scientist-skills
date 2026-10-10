# Status: Data Scientist Skill Set

Last updated: 2026-09-27, end of session 1. Work paused by Apoorva ("save and stop, we will start later").

## State

| Item | Status | Where |
|---|---|---|
| Design agreed (scope, 10 skills, gates, two-phase comparison, reasoning rule) | Done | `README.md` (architecture), `ds-skill-inventory.md` (stage-by-stage inventory) |
| Synthetic test datasets with planted defects + answer keys | Done | `evals/make_datasets.py`, `evals/datasets/`, `evals/answer_keys/` |
| Eval prompts | Done | `evals/evals.json` |
| Grading assertions per eval | Drafted | `data-scientist-workspace/iteration-1/<eval>/eval_metadata.json` |
| Baseline runs without any skill (RED evidence) | Partial, stopped | `data-scientist-workspace/iteration-1/<eval>/without_skill/outputs/` (churn: EDA + leakage checks + model-compare script, no report; house: EDA + anomaly checks + model-comparison output, no report; forecast: EDA only) |
| Verified method inventories (research) | Not produced | agents were stopped before writing; `research/` does not exist |
| Skills written | 0 of 10 | none yet |
| Installed into `~/.claude/skills` | No | see README "Install" |

## Decisions made, with reasons

1. Scope is tabular + time series only. NLP/CV are covered by existing Hugging Face skills.
2. One orchestrator (`data-scientist`) plus nine step skills, called by name from the orchestrator. Apoorva wants
   each step deep and standalone; the orchestrator calling them by name avoids relying on spontaneous triggering.
   `ds-forecasting` was added as its own skill because the time-series menu is large and maps to one test case.
3. Every gate must produce an artefact (a table, a number, a plot), not a yes. Rubber-stamped gates are how steps get missed.
4. Each stage carries forward the best 2-3 candidates; the modelling stage runs a joint conditional search over them
   (Optuna with pruning). Correctness gates (leakage, split design, sentinel cleanup, test-set discipline) are never searched over.
5. Selection guards: repeated seeds, one-standard-error rule, a locked holdout for the search, test set scored once.
6. Reasoning is mandatory both ways: every technique applied or skipped gets "considered / applied because / skipped because".
7. No compute tier is assumed. Choose by data size and problem; if a method needs heavy compute, state it and ask.
8. Run agents sequentially, not in parallel, to conserve tokens. Use a cheaper model (Sonnet) for test-run agents; keep the
   session model for writing the skills.
9. Existing skill `ml-experiment-measurement` already covers noise-floor and ablation discipline; point to it, do not copy.

## Resume plan (in order, one step at a time)

1. Read the partial baseline outputs (the `.py` and `_output.txt` files) as evidence of what gets missed without a skill.
   Decide whether to finish one baseline (churn) or proceed on the partial evidence.
2. Write `data-scientist/SKILL.md` (orchestrator): framing, stage order, gates with required artefacts, decision-log template,
   calls to step skills by name, test-once rule, report template.
3. Write step skills one at a time, each with `references/` from a verified inventory. Verify only risky facts
   (latest versions, 2025-2026 models) with a few targeted lookups per skill rather than full research agents.
   Order: ds-data-audit, ds-validation-design, ds-method-comparison, ds-missing-values, ds-outliers,
   ds-feature-engineering, ds-model-selection, ds-forecasting, ds-evaluation.
4. Run one with-skill test at a time (churn first), grade against `evals/answer_keys/` using the assertions, fix, rerun.
5. Install via junctions (README) and tune trigger descriptions.

## Working rule

At most 2 tasks or agents run at the same time. Everything else waits in the queue below.
Pull the next item only when a running one finishes. Short update to Apoorva after each step.

## Queue

| # | Task | State |
|---|---|---|
| 1 | Read partial baseline outputs; list what was missed without a skill | Done (session 2), findings below |
| 1b | Fix 3 answer-key errors found by the baselines; regenerate data; verify | Done (session 2), verified |
| 2 | Write `data-scientist` orchestrator SKILL.md + references/templates.md | Done (session 2): 198 lines, frontmatter valid |
| 3 | Write `ds-data-audit` + verify its risky facts | Done (session 2): SKILL.md, references/methods.md, scripts column_audit.py + leak_screen.py tested on all 3 datasets; tools/validate_skills.py added |
| 4 | Write `ds-validation-design` | Done (session 2): SKILL.md, scripts make_split.py (5 schemes, gap, test hash) + cv_folds.py (fold loader, test-untouched check), tested on all 3 datasets. Generator fix: resales kept inside the data window; audit flags FUTURE_DATES |
| 5 | Write `ds-method-comparison` | Done (session 2): SKILL.md, references/search-methods.md, scripts screen.py (paired Nadeau-Bengio SE, seed-averaging) + joint_search.py (conditional TPE, distinct finalists, 1-SE pick, holdout check), tested end to end on churn |
| 6 | Write `ds-missing-values` | Done (session 2): SKILL.md, references/methods.md (maintenance status from PyPI), scripts/missing_profile.py tested on churn + house (matches planted mechanisms) |
| 7 | Write `ds-outliers` | Done (session 2): SKILL.md, references/methods.md (PyOD 3.6.6 families, sklearn classes import-checked), scripts outlier_profile.py + outlier_transformers.py tested on churn + house |
| 8 | Write `ds-feature-engineering` | Done (session 2): SKILL.md + references/methods.md; versions from PyPI, class names confirmed from wheel source (category_encoders 2.11.1, feature_engine 1.9.4) |
| 9 | Write `ds-model-selection` | Done (session 2): SKILL.md + references/methods.md with live TabArena snapshot (2026-09-22 data), foundation-model licence/limit checks, all class names import-checked |
| 10 | Write `ds-forecasting` | Done (session 2): SKILL.md + references/methods.md with live fev-bench weekly snapshot (2026-09-18 data), licences via GitHub API, class names from wheel source; scripts/ts_backtest.py tested on store sales (gap typing, per-horizon MASE, origin spread) |
| 11 | Write `ds-evaluation` | Done (session 2): SKILL.md, references/methods.md + model-card.md, scripts/final_eval.py tested end to end on churn (test AUC 0.722 vs honest ceiling 0.73; shuffled-label check 0.099 vs 0.087 base rate) |
| 12 | Churn on Sonnet: one with-skill run + one fresh no-skill run as a pair (2 concurrent max), grade both | Done: no-skill 14/16 (30 min, 187k tokens); with-skill 16/16 (54 min, 330k tokens) but CONTAMINATED (skills quoted churn answers). Same predictive quality (test AUC 0.716 vs 0.720) |
| 12b | Fixes from iteration 1: decontaminate all skills (neutral loan-default examples), paired-SE keep rule (noise floor only for unpaired numbers), noise_floor_sd = sd of CV mean, time budget per stage, not-installed package rule, stateless cleaning as first pipeline step, outliers: screen none vs FarValueNuller when unverifiable, evals description moved to evals/README.md | Done (session 2) |
| 12c | Churn with-skill rerun on decontaminated skills (iteration-2), 25 min budget | Done: 16/16 clean, 13.6 min, 182k tokens (faster and cheaper than no-skill). Found leaks independently; kept last_invoice under ASSUMED timing (correct). Missed usage entry errors because it re-implemented outlier checks instead of using the bundled script |
| 12d | Fixes from iteration 2: bundled scripts are the default (log if replaced); noise floor measured in the primary metric's units | Done (session 2) |
| 13 | House prices on Sonnet: no-skill run, then with-skill run after skill fixes | No-skill 8/13 (15 min); with-skill (iteration-2) 13/13 (25 min, 254k tokens) |
| 13b | Fixes from house run: fresh_copy (CatBoost clone fallback) in screen/joint_search/final_eval; make_split --drop-overlap-groups; compact mode for short budgets | Done (session 2), tested |
| 13c | Benchmark summary | Done: data-scientist-workspace/benchmark.md (clean totals: without 32/41, with 40/41) |
| 14 | Store sales on Sonnet: no-skill run, then with-skill run after skill fixes | No-skill 10/12 (14 min). With-skill (iteration-2) 11/12 (9.7 min, 160k tokens; test WAPE 0.093 vs seasonal naive ~0.15-0.17): zeroed calendar holidays in the forecast, cut audit scripts believing it was over budget. Local: statsforecast installed; MAPIE, fairlearn, tabpfn, tabicl, evidently not |
| 14b | Fixes from store run: backtest passes h; break_report (season-aligned; flags exactly S07 shift + S19 intermittent on test data); future calendar covariates generated, never placeholders; intermittent series judged by MASE; time budget measured with a clock; audit + leak scripts are never-cut correctness gates | Done (session 2) |
| 15 | Fix skills from test findings, rerun failing evals | Done. Verification (iteration-3): store 12/12, churn 15/16 (missed age = 0). Fixes from it, each tested on the eval data: composite --id + make_ids; paired_compare; forecasting tool mapping; ZERO_VALUES? flag; compact mode keeps profiling scripts; joint_search finds the skill set when copied; RuleBaseline. See data-scientist-workspace/benchmark.md |
| 16 | Install via junctions into ~/.claude/skills, tune trigger descriptions | Waiting for Apoorva's go-ahead (installing affects every future Claude Code session) |
| 17 | **First real-world run**: full blind run on Kaggle S6E9, 235 min, 11/11 gates, compared against ~50-60 prior experiments on the same data | Done 2026-09-28. Workspace `ml-bench/ev-purchase-s6e9/ds-rerun/`. Score 17 AGREED / 2 DISAGREED / 14 NEW / 5 MISSED. Local test AUC 0.945514, LB 0.94583 vs prior best LB 0.94615 |
| 18 | **Iteration-4 fixes from that run** (13 changes across 8 skills + 2 scripts) | Done 2026-09-28, see `ITERATION-4-FIXES.md`. `tools/validate_skills.py` passes on all ten |
| 19 | **Adversarial review of iteration 4** (Fable): 17 findings, 1 critical (§5b re-carved the test set), 8 major | Done 2026-09-28. All critical/major fixed; run record received a CORRECTIONS block; `column_audit.py` and `joint_search.py` re-verified against the reviewer's failing cases; pre-existing eval-name hygiene hits in ds-forecasting/cv_folds/make_split removed |

## Task 1 findings: what the no-skill baselines did (partial runs, strong model, old data)

Got right: dedupe, sentinels, category normalisation, impossible values, a leak audit (with/without each
suspicious column), preprocessing inside pipelines, repeated stratified CV with spread, grouped CV by
property_id, log target, rolling-origin backtest with 10 origins, dummy and business-rule baselines,
Optuna on train only, test held back with bootstrap CI.

Missed or weak (these become gates in the skills):
- Neither run finished inside its time budget or wrote a report. Slow model families were fitted before a
  cheap screen showed logistic regression was already best. -> compute budgeting + cheap screen first.
- No decision log; nothing skipped was given a reason. -> mandatory considered/applied/skipped block.
- One method per stage; never 2-3 carried forward. -> carry-forward + joint search.
- Churn: usage outliers never examined; balanced logistic probabilities miscalibrated (Brier ~0.22).
- House: nobody asked when tax_assessed_value was recorded relative to the sale. -> record-timing question
  for every strong feature, not only an importance check.

Answer-key errors found and fixed (task 1b):
- house: tax_assessed_value was generated from the final price incl. sale noise (a leak). Now generated from
  the expected price with independent 10% error. Verified: best in-sample OLS residual sd 0.118 vs 0.12 noise.
- churn: last_invoice missingness was generated from the label (a leak). Now driven by pre-outcome risk p.
  Verified: missing 30% for churners vs 16.5% otherwise.
- store: using inventory_end at the forecast origin is legitimate; only same-week use is a leak. Assertion reworded.

Consequence: the partial baselines ran on old data and a strong model. Fresh no-skill baselines are needed on
Sonnet, paired with the with-skill runs (queue items 12-14). Weaker models are where the skills should show.

## Notes

- Paused subagents from session 1 will not survive into a new session; research must be redone, one area at a time.
- The three long-running `src/*.py` Python processes seen on this machine belong to another project of Apoorva's. Do not touch them.
- Latest scikit-learn on PyPI is 1.9.1 (2026-09-10) but this machine has 1.7.2. Skills state sklearn facts "on 1.7.2";
  recheck experimental flags (IterativeImputer, HalvingRandomSearchCV) after any upgrade.
- Local environment verified 2026-09-27: Python 3.13.9, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, lightgbm 4.7.0,
  xgboost 3.4.1, optuna 5.0.0.
