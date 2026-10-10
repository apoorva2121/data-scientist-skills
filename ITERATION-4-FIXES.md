# Iteration 4 — fixes from the EV S6E9 real-world run

Date: 2026-09-28. Source: a full blind run of the orchestrator on Kaggle Playground S6E9
(`ml-bench/ev-purchase-s6e9/ds-rerun/`), 235 minutes, all 11 gates passed, compared afterwards
against ~50–60 prior experiments on the same data.

This is the first time the skill set was run on a **real** dataset with an independent prior
analysis to check it against, rather than on a synthetic eval with a planted answer key. Every
fix below traces to something that actually went wrong or was actually missed in that run.

## What the run got right (do not regress these)

It found the single decisive lever independently — a target encoding of the quasi-identifier
column, +0.004 AUC, eight times the noise floor — in 4 hours against ~50–60 prior experiments.
It reached parity on equal data (0.945698 against the prior work's 0.945650 / 0.94579). It caught
two false-positive sentinel flags, correctly refused to drop three DOMINANT columns that the
skill's own rule said to drop, and produced calibration, slices, a learnability verdict and a
grouped diagnostic that the prior work did not have.

Crucially the finding came out of the **Stage 1 audit**, not the search. The gates work.

## The thirteen fixes

### 1. Provenance — a whole missing question (orchestrator, Stage 0b)

**What happened.** The run discovered at Stage 1 that the competition rows were replicated from a
10,000-row source table, and correctly concluded income was a quasi-identifier. It then built
**zero features from that table** — no membership, no count, no distance-to-nearest — because no
stage ever asked it to. The prior work had `dist_to_orig_income` in its baseline.

It also never asked whether the generating rule could be recovered. The prior work fitted a probit
(σ = 0.851) and computed an analytic ceiling of 0.93689, which reframes the entire problem: every
score above ~0.9378 is artefact exploitation, not prediction.

**Fix.** New **Stage 0b: Provenance** in `data-scientist/SKILL.md` with three questions — what
other files came with the data, how was the data made, and if it was generated what is the
achievable ceiling. Plus a step **1b** in `ds-data-audit` requiring every other file in the data
directory to be opened and related to the modelling table, and a features-from-other-files table
in `ds-feature-engineering`.

**Evidence it matters.** The run's grouped diagnostic later landed at 0.936222 — within 0.0007 of
the prior work's analytically derived 0.93689, by a completely different method. Two independent
routes to the same boundary. Without Stage 0b, only one of them exists.

### 2. Menu coverage is unenforced (orchestrator, decision log)

**What happened.** Frequency encoding **is** in `ds-feature-engineering`'s encoding table. The run
never screened it. Measured afterwards: +0.000106 — small, but it was free and it was in the menu.
The decision-log template says "Considered: every candidate the step skill lists", but nothing
checks that claim, so the log asserted completeness it did not have.

**Fix.** A menu-coverage gate: before closing a stage, walk the step skill's decision table and
confirm every applicable row appears as applied / carried / skipped-with-reason, and write the
coverage check itself into the block.

### 3. One split is not enough (`ds-validation-design`, new section 5b)

**What happened.** Every conclusion in the run rests on one stratified partition. The prior
protocol verifies every conclusion on a second independently seeded split with an accept bar on
both. At 434k rows the paired SE is so small that "consistent across all 5 folds" is nearly
automatic and proves much less than it appears to.

**Fix.** Survivors must be re-screened on a second seeded partition before entering Stage 8.
Split B is a second validation partition, not a second test set.

### 4. The noise floor degenerates at large n (`ds-validation-design`, section 5)

**What happened.** The prescribed `repeat_sd` came back at **1.5e-06** on 434,632 rows, because
repeats reshuffle partitions of the same rows rather than resampling data. Real sampling noise
measured by bootstrap was **0.00050** — 330× larger. Adopting the prescribed value would have
called a 0.0001 difference significant.

**Fix.** A degeneracy check (`repeat_sd < fold_sd / (10·√k)` means it has collapsed) and a
bootstrap-over-rows alternative, with an explicit statement of which question each statistic
answers. The paired SE asks "consistent on *these* rows"; the bootstrap asks "would it replicate
on *other* rows". Carry-forward decisions need the second.

### 5. The test carve-out has an unpriced cost (`ds-validation-design`, new section 5c)

**What happened.** Carving 20% test + 15% holdout removed 35% of the data from model selection.
The identical pipeline scores **0.945206** on the 434,632-row training portion and **0.945698**
cross-validated on all 668,665 rows: **+0.000492**. That is just *under* the run's own 0.00050
noise floor, so by the run's own rule it is a tie - the direction is supported by the Stage 7
learning curve (still rising at full size), the magnitude is plausible rather than established.
Note also that the 0.945698 figure was computed over all rows, test included (see decision_log
corrections #4).

**Fix.** Section 5c prices the carve-out against the Stage 7 learning curve, requires the cost to
be reported next to the test score, and names the cases where nested CV is the better trade —
notably when the delivered model is refit on 100% of the data anyway.

### 6. `column_audit.py` misses density-spike placeholders (`ds-data-audit`)

**What happened.** The script flags sentinels by value pattern (-999, 99999) and **missed both
censoring floors** — income 30000 with 61,605 rows sitting next to a value with 2 rows, and
commute 5.0 with 144,280 rows next to 54. These are the most conspicuous structural features of
the dataset. It also raised two false positives (a monotonic `id` column, and income 99999 ×28
which sits inside its local count distribution).

**Fix, in code.** `_density_spikes()` added to `column_audit.py`. Three criteria, any of which
fires, all requiring the value to hold at least 0.5% of rows and (for the second and third) to be
**isolated** - its immediately adjacent values hold under half its count:

| criterion | fires when | catches |
|---|---|---|
| continuous | count >= 50x the median of the 25 neighbours each side | a floor at 30,000 next to a value holding 2 rows |
| discrete | isolated, share >= 5%, count >= 8x neighbour median | a 20% floor in an integer age column, where every neighbour already holds ~1.5% |
| edge | isolated, at the minimum or maximum, share >= 3%, count >= 3x neighbour median | a 5% top-code, only ~3-4x its neighbours |

Results are ranked by share before truncating to three, so a ceiling is never dropped behind
heaping values. Columns with under 10 distinct values are skipped (every value is common); columns
unique per row are skipped (nothing can spike; this also took the 668k-row `id` column from 38 s to
0.06 s). The isolation test is what separates a placeholder from the natural mode of a skewed
count distribution: a Poisson(3.2) peak is 12x the windowed median but its neighbours hold 94% of
its count, so it is silent, while a zero-inflated column with 25% forced zeros fires.

**Verified, after the review.** The reviewer's six constructed misses (integer floor 20%,
top-code 5%, tenure 15% at 0, percent 10% at 100, continuous 2.3% ceiling, ceiling behind heaping)
all fire; five clean distributions (uniform ints, normal, Likert, Poisson mode, uniform 1-30) are
silent. Real-data regression unchanged: the dataset that defeated the original script reports its
two floors and nothing else; churn shows a real planted floor plus the -999 sentinel from a second
angle; store sales shows zero-inflation; house prices nothing.

### 7. "Timing unknown + DOMINANT → drop" would have destroyed the model (`ds-data-audit`)

**What happened.** Three columns were flagged DOMINANT. Applying the rule mechanically drops them
and takes CV AUC from 0.940585 to 0.604602. The run overrode it on the grounds that all 13 features
ship in the competition's own `test.csv`, so availability at prediction time is established by
construction.

**Status: the override was right and is already documented in the run, but the rule text has not
been changed.** Recommended wording for a future iteration: the precondition should be "not
available at prediction time", and presence in a supplied scoring file should be listed as
admissible timing evidence. Left unpatched here because it needs care — the rule catches real
leaks and should not be weakened casually.

### 8. Encoder hyperparameters were never treated as search dimensions (`ds-feature-engineering`)

**What happened.** The run varied encoding **granularity** (exact, //50, //500) and left smoothing
at `smooth="auto"`. The prior code's docstring states that sklearn's auto degenerates to m≈1 on a
binary target, leaving singletons nearly unsmoothed. The run's own data agreed and it did not
notice: `exact_smooth2` beat `exact` with auto by +0.000142.

Measured afterwards: granularity alone 0.945698, smoothing alone 0.945540. The 0.945895 figure
sometimes quoted as "both together" is candidate G, which also adds commute TE, frequency and
distance-to-source; no run isolated granularity + smoothing, and all differences here are within
the noise floor. Screen both axes; do not assume either covers the other.

**Fix.** A table of encoder knobs — smoothing, granularity, both, `min_frequency` — with the
explicit warning about `smooth="auto"` on binary targets and the note that granularity and
smoothing are not substitutes.

### 9. Early-stopping leakage is not covered (`ds-method-comparison`)

**What happened.** The skill warns against early-stopping on the *test* set. It says nothing about
the far more common case: early-stopping on the **validation fold you are about to score**, which
is the default in nearly every boosting tutorial. The bias is small, consistent across folds, and
therefore invisible to a paired test.

**Fix.** A section on honest early stopping — choose the round count on a slice of the fold's
*training* portion, refit, then score — plus the rule that a shuffled-label canary must be run
with early stopping **off**, or stopping adapts to the permuted labels.

### 10. The search space shrank silently (`ds-method-comparison` + `joint_search.py`)

**What happened, and this is the worst one.** `joint_search.py` only offers an option when its
name is a key of `SLOTS[slot]` **and** the family is in that option's `applies_to`. Both are silent
filters. In the run, `search_space.json` used Stage 4/5 candidate names, and the `applies_to` lists
predated the XGBoost family being added at Stage 7. Result:

```
slot 'missing_values': names not in SLOTS[missing_values] and therefore IGNORED:
    ['none', 'floor_indicator_kept_value', 'floors_nan_native']
slot 'encoding': no entries in search_space.json -> always defaults to 'onehot'
slot 'outliers' x family 'xgb': ZERO options -> silently defaults
```

Two of three slots were **never sampled for any family**. The run reported that they "never left
their no-op settings", implying they were searched and lost. They were never offered.

**Fix.** `joint_search.py` now validates the space before running: it prints every unmatched name,
every slot×family pair with zero options, and the effective space per family, plus a
`--validate-only` flag to check before spending compute. Verified by running it against the exact
`search_space.json` that produced the bug — it reports all three problems.

### 11. Picklability is never tested until the design is locked (orchestrator Stage 10, `ds-evaluation`)

**What happened.** Stage 10 failed three times. A closure could not be pickled; then a lambda
returned by a factory could not be pickled; then the file **saved successfully and could not be
reloaded**, because the function had pickled as `__main__.prep`.

The third is the dangerous one: every check upstream passed, `pipeline.joblib` existed on disk at
1.4 MB, and the handoff was worthless.

**Fix.** A "reload in a fresh process and reproduce a prediction" row in `ds-evaluation`'s sanity
table and in the orchestrator's Stage 10, with the rule that every transformer reaching the
artifact must be a module-level class or function carrying its configuration as instance state —
checked when the first custom transformer is written, not at Stage 10.

### Bonus: a defaults screen ranks library defaults (`ds-model-selection`)

XGBoost was second-worst at defaults (0.944084, outside one paired SE) and **won the Stage 8
search after tuning**. The run carried it against a strict reading of the carry-forward rule and
that judgement paid off. `ds-model-selection` now says so: carry a cheap family past a marginal
defaults deficit, and drop on the defaults screen only when a family is expensive as well as
behind. CatBoost was correctly dropped at 65× the cost for a tie.

## Enforced in code versus written as guidance

This distinction matters more than the count of fixes, because the run's own failures were mostly
failures of discipline, not of knowledge - the information was often already in the skill.

| Fix | Form |
|---|---|
| 6 density spikes | **code** - `column_audit.py` raises the flag; verified on the data that defeated it and regression-tested on 3 eval datasets |
| 10 search-space validation | **code** - `joint_search.py` prints every unmatched name and every zero-option slot, with `--validate-only`; verified against the exact file that caused the bug |
| 1, 2, 3, 4, 5, 8, 9, 11, 12 | **guidance** - prose a future run has to read and follow |

Three of thirteen are mechanisms. The rest rely on the agent actually doing what the skill says, which
is the same failure mode that produced several of these gaps in the first place. The highest-value
next work on this skill set is converting more of column two into column one - in particular the
menu-coverage gate (2), which is currently discipline and could be a script if each skill's menu
were machine-readable.

### 13. Leave-one-out target encoding (`ds-feature-engineering`)

**What happened.** Found while trying to bound remaining headroom. A leave-one-out oracle scored
far above a naive own-label oracle, which was the opposite of the expectation that motivated it.
The mechanism is arithmetic: `(S - y_i)/(n - 1)` is a strictly decreasing function of the row's
own label, so within a group the LOO value *is* the label, inverted.

**Corrected after review.** The first write-up quoted numbers with no script in the repository
behind them and claimed "per-fold wrapping does not fix it". Both were wrong. An honestly wrapped
LOO encoder does not inflate CV - it creates a train/serve mismatch and, in the reviewer's
reproduction, scored **below no encoding at all**. The advice "never use LOO" stands; the reason is
that it hurts, not that it leaks invisibly. The first version's within-group sanity check would
also have flagged the recommended cross-fitted encoder as a leak, because every out-of-fold
estimate is mildly anti-correlated with the own label on *training* rows (benign K-fold bias). The
check now runs on held-out rows, where an honest encoder shows no within-group difference.

**Fix.** Rewritten block in `ds-feature-engineering/SKILL.md`: the arithmetic, the two
consequences, the generalisation to any exclude-self statistic, and the corrected held-out check.

## Review of this iteration

An adversarial review by an independent agent (Fable) found 17 issues in the first version of
these fixes, one critical (a further false positive in the density detector - the natural mode of a
skewed count column - was found by the author's own probe while re-testing and fixed with an
isolation test): the §5b second-split instruction re-carved the test set via
`make_split.py --seed 7` and would have put two thirds of the frozen test rows into training. All
critical and major findings were fixed; the run's own record received a CORRECTIONS block rather
than silent edits. The lesson is the same one the run itself taught: prose written confidently and
never executed is where the errors live.

## Enforced in code versus written as guidance

This distinction matters more than the count of fixes, because the run's own failures were mostly
failures of discipline, not of knowledge - the information was often already in the skill.

| Fix | Form |
|---|---|
| 6 density spikes | **code** - `column_audit.py` raises the flag; verified on the data that defeated it and regression-tested on 3 eval datasets |
| 10 search-space validation | **code** - `joint_search.py` prints every unmatched name and every zero-option slot, with `--validate-only`; verified against the exact file that caused the bug |
| 1, 2, 3, 4, 5, 8, 9, 11, 12 | **guidance** - prose a future run has to read and follow |

Three of thirteen are mechanisms. The rest rely on the agent actually doing what the skill says, which
is the same failure mode that produced several of these gaps in the first place. The highest-value
next work on this skill set is converting more of column two into column one - in particular the
menu-coverage gate (2), which is currently discipline and could be a script if each skill's menu
were machine-readable.

### 13. Leave-one-out target encoding is a worse leak than the naive kind (`ds-feature-engineering`)

**What happened.** Found while trying to bound the remaining headroom after the run. A
leave-one-out oracle, intended as the *less* leaky of two diagnostics, scored **0.980** against an
honest 0.946 - and against 0.949 for the oracle that naively includes the row's own label.
Subtracting your own label leaves the remaining statistic a strictly decreasing function of it, so
the encoding **is** the label inverted. Measured: within the same income value, mean LOO encoding
was 0.197 for `y=1` rows and 0.227 for `y=0` rows.

The skill listed `LeaveOneOut` among `category_encoders` variants with a generic "wrap them per
fold or they leak" caveat. Per-fold wrapping does **not** fix LOO - the leak is within the fold.

**Fix.** An explicit "never use leave-one-out target encoding" block, the generalisation to any
exclude-self group statistic, and a sanity check: inside a fixed group, compare the feature's mean
for positive and negative rows.

## Files changed

| File | Change |
|---|---|
| `data-scientist/SKILL.md` | Stage 0b provenance; provenance row in the Stage 0 table; menu-coverage gate; Stage 10 reload test; 4 new red flags |
| `ds-data-audit/SKILL.md` | Step 1b other files; density-spike section; `DENSITY_SPIKE` verdict row; Gate A additions; 2 new red flags |
| `ds-data-audit/scripts/column_audit.py` | **`_density_spikes()` detector** - the script that missed both floors now catches them |
| `ds-validation-design/SKILL.md` | §5 noise-floor degeneracy + bootstrap; §5b second split; §5c carve-out cost; gate additions; 3 new red flags |
| `ds-feature-engineering/SKILL.md` | Encoder-hyperparameter table; features-from-other-files table; frequency encoding emphasised; **leave-one-out leak warning** |
| `ds-method-comparison/SKILL.md` | Honest early stopping; search-space validation; 2 new red flags |
| `ds-method-comparison/scripts/joint_search.py` | Space validation with warnings + `--validate-only` |
| `ds-model-selection/SKILL.md` | Defaults screens rank library defaults, not families |
| `ds-evaluation/SKILL.md` | Reload-in-a-fresh-process sanity check |

`tools/validate_skills.py` passes on all ten skills after the changes.

## Not fixed, deliberately

- **The DOMINANT-drop rule wording** (item 7). It catches real leaks; weakening it needs more
  evidence than one competition where the answer happened to be "keep".
- **`summarize()` rounding to 4 decimals.** Documented in the run as a defect; changing the output
  precision of a shared helper is a wider decision than this iteration should take alone.
- **A general "screen the whole menu" automation.** The coverage gate is currently a discipline,
  not a script. A real checker would need each skill's menu in machine-readable form, which is a
  larger change worth doing once rather than hastily.

## The honest limit of this iteration

These thirteen fixes come from **one dataset**. Several — the noise-floor degeneracy, the carve-out
cost, the defaults-screen caveat — are large-n phenomena that would not have surfaced on a
5,000-row problem, and a small-data run would likely expose a different set. "The skill should
miss nothing" is not reachable as an absolute; what is reachable is that it should not miss a
*category of question*, and that omissions should be caught by a mechanism rather than by luck.
Items 2 and 10 are the ones that generalise furthest, because they are checks rather than content.
