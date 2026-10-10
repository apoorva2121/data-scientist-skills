# Benchmark: with vs without the skills (Sonnet test agents, 2026-09-27)

Grading: inline against each eval's `eval_metadata.json` assertions, from the runs' output files.
Answer keys in `evals/answer_keys/` were never shown to test agents.

| Eval | Run | Checks passed | Headline result | Time | Tokens |
|---|---|---|---|---|---|
| Churn | without skills | 14 / 16 | test ROC-AUC 0.720, precision@10% 25.8% | 30.2 min | 187k |
| Churn | with skills, iteration 1 (contaminated) | 16 / 16 | ROC-AUC 0.716, precision@10% 22.5% | 53.7 min | 330k |
| Churn | with skills, iteration 2 | 16 / 16 | ROC-AUC 0.718, precision@10% 26.7% [19.2, 35.0] | 13.6 min | 182k |
| House prices | without skills | 8 / 13 | MAE $12,642 (random grouped split) | 15.3 min | 132k |
| House prices | with skills, iteration 2 | 13 / 13 | MAE $13,650 [12,490, 14,918] (chronological split), 90% interval coverage 88.7% | 25.0 min | 254k |
| Store sales | without skills | 10 / 12 | backtest WAPE 8.6% | 14.2 min | 139k |
| Store sales | with skills, iteration 2 | 11 / 12 | test WAPE 9.3% vs seasonal naive 15-17% | 9.7 min | 160k |
| Store sales | with skills, iteration 3 (verification) | 12 / 12 | test WAPE 12.2% [10.0, 14.8], MASE 0.99 vs seasonal naive 1.14 | 11.4 min | 176k |
| Churn | with skills, iteration 3 (verification) | 15 / 16 | ROC-AUC 0.719, precision@10% 25.8% [19.2, 32.5] | 14.6 min | 186k |

Totals, first clean skill run per eval: without skills 32 / 41 (78%); with skills 40 / 41 (98%).
Verification runs (iteration 3): 27 / 28. Every miss across all skill runs led to a fix below.

## What the skills changed

- **Subtle planted defects** found only with the skills: small square-footage errors, the ordered
  condition column, target-derived leak confirmed arithmetically (all house prices).
- **Comparison discipline**: fold or origin spread and trivial plus business-rule baselines in every
  skill run; missing from two of three no-skill runs.
- **Reasoning both ways**: every skill run logged what it skipped and why; no no-skill run did.
- **Production readiness**: stateless cleaning packaged into the saved pipeline; raw-row, duplicate
  and directional sanity tests; the first churn skill run caught a raw-row bug that the no-skill
  run's saved model also has.
- **Honest estimates**: chronological split for house prices (the honest estimate for future
  listings under drift), measured interval coverage, named worst slices, ASSUMED lines for the owner.

## What the skills did not change

- **Predictive accuracy on these datasets** is the same within noise. The signals are simple and a
  strong model finds them unaided. The skills make results more complete and more trustworthy, not
  automatically more accurate.
- **Cost**: similar on churn and store sales after the time-budget fix; about 1.6x time and 1.9x
  tokens on house prices.

## Failures that drove fixes

| Found in | Problem | Fix |
|---|---|---|
| Churn it1 | Skills quoted eval answers (contamination) | Neutral loan-default examples; evals description moved out of README; hygiene rule |
| Churn it1 | Noise floor used as a paired tolerance, per-fold sd | Paired corrected SE drives keep; noise floor = sd of the CV mean, for unpaired numbers only |
| Churn it1 | 54 min against a 20 min budget | Time shares per stage; later: measured with a clock; compact mode |
| Churn it1 | Saved pipeline could not clean raw rows | Stage 1 rules become the first pipeline step |
| Churn it2 | Re-implemented outlier checks, missed entry errors | Bundled scripts are the default; replacing one must be logged |
| Churn it2 | Noise floor measured in the wrong metric | Measure in the primary metric's units |
| Store it2 | Holidays zeroed in the forecast | Calendar covariates generated for the horizon, never placeholders |
| Store it2 | Cut audit scripts believing it was over budget (it was not) | Clock-measured time; audit and leak scripts are never-cut gates |
| Store it2 | Backtest did not pass h; level shift in S07 not found | h passed; `break_report` (season-aligned) flags exactly S07 and S19 |
| House it2 | CatBoost with cat_features fails sklearn clone | `fresh_copy` falls back to deepcopy in all helpers |
| House it2 | Dropping overlapping entities was manual | `make_split.py --drop-overlap-groups` |

| Store it3 | make_split needed one id column for panels; no paired comparison for forecasts | Composite `--id a,b` + `make_ids`; `ts_backtest.paired_compare`; forecasting tool mapping in the orchestrator |
| Churn it3 | Kept age = 0 as real; skipped profiling scripts in compact mode | `ZERO_VALUES?` audit flag (29/29, no false flags); both-ends range rule; compact mode lists scripts that always run |
| Churn it3 | Copied template lost its imports; rule baseline needed a workaround | `joint_search.py` locates the skill set; `RuleBaseline` helper |

Iteration 3 confirmed the store fixes (calendar holidays, break report, clock) and the churn fixes
for bundled scripts in general. The iteration-3 fixes themselves (last three rows) are tested on the
evaluation data but not yet exercised by a fresh end-to-end skill run.
