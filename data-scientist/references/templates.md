# Templates

Copy these into the project workspace at Stage 0. Replace every `<...>`.

## Contents
1. problem_statement.md
2. decision_log.md
3. search_space.json
4. report.md

---

## 1. problem_statement.md

```markdown
# Problem statement: <project name>
Date: <YYYY-MM-DD>   Author: <name>   Status of assumptions: <n> ASSUMED lines open

| Field | Value | Source |
|---|---|---|
| Unit of analysis | <one row = ...> | <user / data / ASSUMED> |
| Target | <column, exact definition, observation window> | |
| Prediction moment | <when the score is produced> | |
| Decision fed | <who acts, what action, capacity limits> | |
| Problem type | <type> -> path: <supervised / forecasting / unsupervised> | |
| Primary metric | <metric> because <cost of errors / capacity> | |
| Secondary metrics | <calibration, coverage, per-slice ...> | |
| Baseline to beat | <current process or rule, and its score if known> | |
| Success threshold | <number> | |
| Constraints | <latency, interpretability, regulation, fairness, retraining> | |
| Data size | <rows, cols, positives or series count, history length> | |
| Time and compute | <wall-clock budget, CPU/GPU, RAM> | |

Needs ML at all? <yes/no and why>
```

## 2. decision_log.md

One block per stage, appended in order. Use the exact headings so a reader can scan them.

```markdown
## Stage <n> - <name> - GATE <PASSED | BLOCKED: reason>
Considered: <every candidate the step skill lists that is relevant to this problem type>
Applied (fixed): <correctness steps, not searched>
Carried forward: <name (proxy, metric mean +/- sd)>, ...
Skipped: <name> - <reason tied to this data, family, size or constraint>; ...
Evidence: <files / tables>
Open risks: <assumptions to confirm, things to watch>
```

Reason quality bar: a reason names a number from this data, a property of the chosen model
family, a size limit, or a stated constraint. "Not needed", "standard practice" and "for
simplicity" are not reasons.

## 3. search_space.json

Written from Stage 4 onward. Consumed by `ds-method-comparison` at Stage 8.

```json
{
  "cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 3, "groups": null, "seed": 42},
  "primary_metric": "neg_log_loss",
  "noise_floor_sd": 0.004,
  "noise_floor_definition": "sd of the CV mean across repeats (repeat_sd), not the per-fold sd",
  "fixed_steps": [
    "annual_income: 99999 -> NaN",
    "drop columns: recovery_amount, collections_flag (recorded after default)"
  ],
  "stages": {
    "missing_values": [
      {"name": "median_indicator", "applies_to": ["linear", "gbdt"], "proxy_scores": {"linear": [-0.412, 0.006], "gbdt": [-0.407, 0.007]}},
      {"name": "native_nan", "applies_to": ["gbdt"], "proxy_scores": {"gbdt": [-0.405, 0.007]}}
    ],
    "outliers": [
      {"name": "none", "applies_to": ["linear", "gbdt"], "proxy_scores": {"linear": [-0.412, 0.006], "gbdt": [-0.405, 0.007]}},
      {"name": "log1p_income", "applies_to": ["linear"], "proxy_scores": {"linear": [-0.409, 0.006]}}
    ],
    "features": [],
    "models": []
  }
}
```

`proxy_scores` values are `[mean, sd]` across CV folds and repeats. `noise_floor_sd` is the
standard deviation of the **CV mean** across repeats or seeds (about the per-fold sd divided by the
square root of the number of folds), because it is compared with differences between means.
`applies_to` drives the
conditional search: a candidate is only sampled when the chosen model family is in its list.

## 4. report.md

Write the skeleton at Stage 0. Fill each section when its gate passes. Mark unfilled sections
`NOT REACHED` so an interrupted run is still honest.

```markdown
# <Project>: model report
Status: <gates passed, e.g. 0-7 passed, 8 in progress>

## 1. Answer first
<2-4 sentences: what the model does, how good it is on the untouched test set with an interval,
how that compares with the baseline, whether it meets the success threshold.>

## 2. Problem
<Summary of problem_statement.md; list open ASSUMED lines.>

## 3. Data
<Rows, columns, period. Cleaning applied with row counts. Leak verdicts. Missingness and
outlier findings.>

## 4. Validation design
<Scheme, why, what was rejected and why. Test set size. Noise floor.>

## 5. Decisions
<One line per stage: what was applied, carried forward, skipped, with the reason. Link the log.>

## 6. Results
| Model / pipeline | CV metric (mean +/- sd) | Locked holdout | Test (95% CI) |
|---|---|---|---|
| Trivial baseline | | | |
| Business rule | | | |
| Simple model | | | |
| Final pipeline | | | |

## 7. What drives the prediction
<Global importance, direction of effects, sanity against domain knowledge.>

## 8. Where it fails
<Worst slices, error analysis, calibration, known failure modes.>

## 9. Limitations and risks
<Assumptions, drift risks, data gaps, fairness notes.>

## 10. Next steps
<Ranked by expected gain: data, labels, features, then modelling.>
```
