# Data audit: methods and tools

Versions and release dates read from PyPI's JSON API on 2026-09-27. API names confirmed against
the package description or the project's API reference the same day. Recheck before pinning.

## Bundled scripts (no extra dependencies beyond pandas, numpy, scipy, scikit-learn)

| Script | Stage | What it finds |
|---|---|---|
| `scripts/column_audit.py` | 1 | Inferred type, missing share, cardinality, sentinel codes (numeric, string, date), spelling variants, non-numeric tokens, rare negatives, extreme max, values beyond 3 IQR, out-of-range years, ID-like and group-key columns, key uniqueness and conflicting duplicates |
| `scripts/leak_screen.py` | 3 | Univariate strength per column (out-of-fold target means for categoricals, grouped folds if given), near-perfect and dominant columns, significant informative missingness (Hanley-McNeil test for binary), target-derived columns via trimmed log-log fits, row-order signal |

Tested 2026-09-27 on the three evaluation datasets in `evals/`. Every planted structural defect and
leak was flagged.

## Profiling (optional, for a human-readable overview)

| Tool | pip name | Latest (released) | Use when | Caveats |
|---|---|---|---|---|
| ydata-profiling `ProfileReport` | `ydata-profiling` | 4.18.4 (2026-04-22) | One HTML report of distributions, correlations, missingness | Slow on wide or large data; use `minimal=True` or a sample |
| skrub `TableReport` | `skrub` | 0.10.1 (2026-09-01) | Fast interactive table summary inside notebooks | Pre-1.0; check API on upgrade |
| Sweetviz | `sweetviz` | 2.3.3 (2026-04-11) | Side-by-side comparison of two datasets (train vs test) | Report-oriented, not scriptable checks |
| missingno | `missingno` | 0.5.2 (2023-02-26) | Missingness matrix and co-occurrence plots | Last release over three years ago; stable but unmaintained |

## Data validation (turn Stage 1 rules into checks that run on every new batch)

| Tool | pip name | Latest (released) | Use when |
|---|---|---|---|
| pandera | `pandera` | 0.33.1 (2026-09-01) | Schema and value-range checks as code, close to pandas |
| Great Expectations | `great-expectations` | 1.23.2 (2026-09-25) | Team-level data contracts with documentation and history |
| Deepchecks (tabular) | `deepchecks` | 0.19.1 (2024-12-15) | Ready-made suites for integrity, leakage and train-test drift | Release cadence has slowed; check before adopting |

Write the Stage 1 verdicts (valid ranges, sentinel codes, canonical levels) as a pandera schema at
Stage 10, so the production pipeline rejects the same problems automatically.

## Label quality

| Tool | pip name | Latest (released) | API | Use when |
|---|---|---|---|---|
| cleanlab | `cleanlab` | 2.9.0 (2026-01-13) | `cleanlab.filter.find_label_issues(labels, pred_probs)`, `Datalab` | Suspected label noise; needs out-of-fold predicted probabilities |

## Leakage taxonomy (what to look for, beyond single strong columns)

1. **Post-outcome columns**: status, reason, resolution, final amounts.
2. **Target-derived arithmetic**: ratios and products involving the target.
3. **Future information via aggregates**: lifetime, rolling or group statistics computed over the full history.
4. **As-of join failures**: dimension tables joined on their current state instead of their state at the prediction moment.
5. **Entity leakage**: the same customer, patient, device or property on both sides of the split.
6. **Duplicate leakage**: exact or near-duplicate rows across the split.
7. **Preprocessing leakage**: any statistic fitted on rows that include the test set.
8. **Order and ID leakage**: files sorted by outcome, IDs issued at the time of the outcome.
9. **Window overlap**: feature windows extending past the prediction moment into the label window.
10. **Evaluation feedback**: repeated looks at the test set steering decisions.

## Adversarial validation

Classifier distinguishing two samples; cross-validated AUC near 0.5 means indistinguishable.
Uses: train vs test (split sanity), train vs recent data (drift), train vs serving extract
(train-serving skew). Inspect the separating model's top features; drop or fix only with a
reason, since removing genuinely predictive shifted features also costs accuracy.
