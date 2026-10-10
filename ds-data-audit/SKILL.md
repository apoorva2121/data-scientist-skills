---
name: ds-data-audit
description: Use when first receiving a tabular or time-series dataset for modelling, before any split or model is fitted; when checking data quality - duplicates, sentinel codes such as -999 or "N/A", inconsistent category spellings, impossible values, ID columns, mixed units; when hunting target leakage or a feature that seems too predictive; when a score looks too good to be true or a model collapses in production; or when exploring feature-target relationships.
---

# Data Audit

## Overview

Two parts, run at two different moments:

- **Part A (Stage 1), whole file:** structure audit and stateless cleaning. Runs before the split
  because duplicates, keys and dates decide how the split must be made.
- **Part B (Stage 3), training rows only:** leakage hunt and relationship EDA. Runs after the test
  set is carved, because looking at feature-target relationships on test rows leaks test
  information into feature choices.

The single most valuable question in this skill: **when is each value written, relative to the
moment the prediction is made?** Predictive strength alone cannot tell a leak from a good
feature; record timing can.

## Part A: Structure audit and stateless cleaning (Stage 1)

**1. Run the audit script.** It reads everything as raw strings, so sentinels and spelling variants
are not hidden by type inference.

```bash
python scripts/column_audit.py data.csv --target <target> --key <id_col[,date_col]> --out column_audit.csv
```

**1b. Open every other file in the data directory.** Not just the one you were told to model.
For each: how many rows, what is the key, and how does it relate to the modelling table? A
reference table, a dictionary, a prior extract or the "original" dataset a derived table was built
from each carries features (membership, count, distance to nearest, rank, partial match) that go
to Stage 6, and each can settle Stage 1 questions the modelling table alone cannot.

Two checks that repay the minute they cost:
- **Does the modelling table's value distribution trace back to the other file?** Compare the
  high-count values against the reference values. If the frequent values are all present there and
  the rare ones are not, rows were generated *from* that file - which means rows are not
  independent, and the linking column is a quasi-identifier. Hand that straight to Stage 2.
- **Do sentinels, floors and nulls behave the same in both?** A floor that exists in the reference
  file *alongside* genuine nulls is censoring, not a missing code. The modelling table alone
  cannot distinguish those two, and the difference changes Stage 4.

**2. Establish the grain.** Write down what one row is. Confirm the key is unique. For duplicates:
exact copies are dropped; rows sharing a key with different values need a rule (latest timestamp,
most complete, source priority), written in the cleaning log.

**3. Resolve every flag.** Add a `verdict` column to `column_audit.csv`. Every flagged row gets one.

| Flag | Meaning | Stateless action here | Handed to |
|---|---|---|---|
| `SENTINEL?`, `STRING_SENTINEL?`, `DATE_SENTINEL?` | A code standing in for missing | Confirm from the distribution or dictionary, then convert to null | Stage 4 |
| `SPELLING_VARIANTS:n` | Case, whitespace or separator variants | Map to canonical levels; print the map | - |
| `NON_NUMERIC_TOKENS` | Units, commas, currency in a number | Parse, or null if unparseable | - |
| `YEAR_OUT_OF_RANGE`, `FAR_OUT_3IQR`, `EXTREME_MAX`, `RARE_NEGATIVES` | Possibly impossible, possibly extreme | Impossible by a domain rule: null it. Merely extreme: leave it | Stage 5 |
| `ZERO_VALUES?:n` | A few zeros far below every other value | Often a disguised missing code (age 0, weight 0): null it if 0 is impossible for the quantity | Stage 4 |
| `DENSITY_SPIKE@v:n` | A value holding far more rows than its neighbours in sorted order - a censoring floor, a clipping ceiling, an imputed default, or zero-inflation | **Decide which of the four it is; the action differs.** Censoring: keep the value, add an at-floor indicator, and never let it contaminate a smooth or binned encoding over that column. Imputed default: null it (Stage 4). Zero-inflation: a modelling choice, not a cleaning one (Stage 7 target family). **Use the other files from step 1b to tell censoring from imputation** - a floor that coexists with genuine nulls in the source is censoring | Stages 4, 5, 6 |

| `ID_LIKE` | Unique per row | Exclude from features, keep as key | Stage 2 |
| `GROUP_KEY?` | Entity repeating across rows | Candidate group for a grouped split | Stage 2 |
| `CONSTANT` | One value | Drop | - |
| `HIGH_CARDINALITY`, `RARE_LEVELS` | Many or sparse levels | Note only | Stage 6 |
| `MISSING:x%` | Nulls present | Note only; mechanism is diagnosed later | Stage 4 |

Check **both ends** of every range rule: an upper limit alone ("age over 120") misses impossible
low values such as zeros.

Also check what the script cannot see: mixed units (a bimodal column at x100 or x1000), impossible
combinations (end date before start date, children older than parents), and columns that start
being filled at some date (a collection change).

**4. Types.** Parse dates with an explicit format. Mark ordered categoricals as ordered. Keep IDs as strings.

**5. Target.** Compare the column with the definition in `problem_statement.md`. Record class
balance or distribution and skew. Classes with fewer than about 20 rows: merge, or flag that they
cannot be evaluated. Rows with a missing target: find out why before dropping them. Missing
because the outcome window has not closed yet (censoring) calls for a cutoff date or survival
methods, not deletion.

**6. Time coverage.** Date range, rows per period, gaps, and the target rate per period.

**7. Cleaning log.** One line per step: rule, rows affected, rows remaining. Then check that the
removed rows are not concentrated in one class, segment or period.

Nothing learned from the data happens here: no imputation, capping, encoding or scaling.

**Write the stateless rules once, as code that production will reuse.** Put them in one function
(for example `clean_raw(df) -> df`) and use it both here and, wrapped in a
`FunctionTransformer(clean_raw)`, as the first step of the final pipeline. Cleaning applied only in
a notebook is cleaning the production model never sees.

**Density spikes.** `column_audit.py` raises `DENSITY_SPIKE@v` for any value carrying at least
0.5% of rows and 50x the median count of its neighbours in sorted order. This is the check that
catches a floor or placeholder **looking like an ordinary number**, which no sentinel pattern will
match: a floor at 30,000 holding 9% of rows, whose next distinct value holds two, matches nothing
in the -999 / 99999 family and is otherwise invisible.

Columns with fewer than 10 distinct values are skipped deliberately - there every value is common,
so the ratio carries no information and the check would fire on ordinary Likert levels.

The flag says a pile-up exists; it cannot say why. Resolving it into censoring / imputation /
zero-inflation is the verdict you write, and step 1b's other files are usually what settles it.

**Gate A** (all must exist):
- every other file in the data directory opened and described, with its relationship to the
  modelling table (step 1b)
- density-spike check run on every continuous column, with a verdict on any pile-up
- `column_audit.csv` with a verdict for every flagged column
- `cleaning_log.md` with row counts per step
- target summary
- candidate keys, group columns and time columns handed to Stage 2
- decision-log block

## Part B: Leakage hunt and relationship EDA (Stage 3, training rows only)

**1. Run the leak screen** on the training portion, with IDs excluded and the Stage 2 groups passed in:

```bash
python scripts/leak_screen.py train.csv --target <target> --task binary|multiclass|regression \
    --exclude <id cols> [--groups <group col>] --out leak_screen.csv
```

Scores are univariate: AUC-type for classification (0.5 = nothing), absolute Spearman for
regression (0 = nothing). Flags: `NEAR_PERFECT`, `DOMINANT`, `INFORMATIVE_MISSINGNESS`,
`DERIVED_FROM_TARGET?` (regression, robust to a few corrupted rows), `ROW_ORDER_SIGNAL`.

**2. Build the leak table** (`leak_table.csv`). Include every flagged column, the top five by score,
and any column whose name suggests an after-the-fact record: status, reason, result, outcome,
closed, cancelled, final, resolved, approved, paid, `*_after`, `days_to_*`.

| column | score | rank | recorded_when | available_at_prediction | verdict | evidence |
|---|---|---|---|---|---|---|

Evidence for timing: the data dictionary, the data owner, timestamps, and fill patterns. A column
that is only filled for positive cases was almost certainly written after the outcome.

| Finding | Verdict | Action |
|---|---|---|
| Written after the prediction moment | leak | Drop. A fixed step, never searched over |
| Arithmetic of the target (`DERIVED_FROM_TARGET?`) | derived | Drop the derived column, keep its honest component |
| Timing unknown and `NEAR_PERFECT` or `DOMINANT` | presumed leak | Drop; report as an open risk; show the score with and without |
| Strong but recorded before the prediction moment | legit | Keep; write the timing assumption down |
| `INFORMATIVE_MISSINGNESS`, missingness arises before the prediction moment | legit | Keep a missing indicator (Stage 4) |

A pair flag (`DERIVED_FROM_TARGET? with X`) lands on both columns. Decide which one was computed
from the outcome; the other is usually a legitimate input.

**3. Check the leak types a column-by-column screen cannot see:**

| Leak type | Check |
|---|---|
| Aggregates over the full history (lifetime averages including the future) | Recompute point-in-time, as of each row's prediction moment |
| Joins without an as-of condition | Max source timestamp per row must be at or before the prediction timestamp |
| Same entity in train and test | Confirm no Stage 2 group straddles the split |
| Near-duplicate rows across the split | Hash rounded feature vectors; count train-test matches |
| Row order or IDs assigned after the outcome | `ROW_ORDER_SIGNAL`; score of the numeric ID |
| Target window overlapping feature windows | Feature window must end at the prediction moment |

**4. Train/serving skew: adversarial validation.** Label training rows 0 and test rows 1 (or a
recent or serving-time sample), fit a small boosted model, read the cross-validated AUC.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
X = pd.concat([train[feats], other[feats]]); d = np.r_[np.zeros(len(train)), np.ones(len(other))]
auc = cross_val_score(HistGradientBoostingClassifier(categorical_features="from_dtype"), X, d,
                      cv=5, scoring="roc_auc").mean()
```

Near 0.5 means similar. Above about 0.6, inspect which features separate them, often time-like
columns or IDs. A shift between training and serving data caps real-world accuracy regardless of
the validation score.

**5. Relationship EDA,** training rows only. Write findings to `eda_notes.md`:

- Shape per feature: skew, zero-inflation, multimodality, heavy tails (hints for Stages 5 and 6).
- Feature against target: target rate or mean per decile for numerics; per level, with counts, for categoricals.
  Note monotonic, U-shaped and threshold relationships.
- Redundancy: pairs with absolute Spearman above 0.9; VIF above 10 matters for linear families.
- Stability: the same relationship per period and segment. A sign flip means drift or Simpson's paradox.
- Label quality (classification, optional): out-of-fold probabilities from a quick model, then
  `cleanlab.filter.find_label_issues`. Review the top suspects; never auto-drop.

**Gate B** (all must exist):
- `leak_table.csv` with a verdict on every row; every `unknown` also listed under open risks
- adversarial validation AUC with the top separating features
- `eda_notes.md` with feature ideas, drop candidates, transform hints, slices to watch at Stage 9
- decision-log block listing each dropped column and why

## Red flags

| Thought | Reality |
|---|---|
| "It's the best feature, keep it" | The best feature is the first one to question for timing |
| "Relationships look the same, quick EDA on all rows is fine" | EDA on test rows shapes features toward the test set |
| "Blank reason just means no reason given" | Filled only for positives means written after the outcome |
| "Impute the -999s later" | -999 is a code, not a value; convert it to null here, stateless |
| "Drop rows with missing target" | First ask whether they are censored rather than missing |
| "No leak flags, so no leaks" | Aggregates, joins and windows leak without any single strong column |
| "The script found no sentinels" | It matches value patterns. A floor at 30,000 with 60k rows next to a value with 2 rows is invisible to it |
| "I only need the training table" | Open every file that shipped with it; they answer questions the table cannot |

## Tools

`references/methods.md` lists profiling, validation, label-noise and drift tools with versions
checked against PyPI on 2026-09-27. Pandas plus the two bundled scripts cover the gates.
