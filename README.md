# Data Scientist Skill Set

Skills for Claude Code that take a tabular or time-series dataset plus a business objective
through to a validated model, without skipping the steps that usually get skipped.

Scope: tabular data and time series. NLP and computer vision are out of scope.

## Why these exist

Most "the model never reached the target" stories trace back to omissions, not to a missing
algorithm: a leaky column nobody questioned, imputation fitted before the split, a random split
on time-ordered data, no baseline, model comparison on a single split, accuracy on an imbalanced
target, tuning against the test set. The orchestrator turns each of those into a gate that must
produce evidence before the next stage starts.

## Layout

| Skill | Stage it owns | What it holds |
|---|---|---|
| `data-scientist` | Orchestrator | Problem framing, stage order, which step skill to call when, the gate at each stage, the decision log, the final test-once rule, hand-off and report templates |
| `ds-data-audit` | Stages 1 and 3 | A: structure audit and stateless cleaning (types, sentinels, spellings, exact duplicates, impossible values). B: leakage hunt with record-timing verdicts and relationship EDA, on training data only |
| `ds-validation-design` | Stage 2 | Split strategies by data structure, CV schemes, carving and freezing the test set, noise floor |
| `ds-missing-values` | Stage 4 | Mechanism diagnosis, the full imputation menu with a decision table |
| `ds-outliers` | Stage 5 | Detection menu, the error / extreme / signal decision, treatment menu |
| `ds-feature-engineering` | Stage 6 | Encoding, transforms, scaling, date and aggregate features, automated generation, selection |
| `ds-model-selection` | Stage 7 | Baselines, candidate families by problem type, imbalance handling, learnability verdict |
| `ds-method-comparison` | Stage 8 + screening at 4-7 | Local screening with two proxy models, joint conditional pipeline search, selection rules that resist noise |
| `ds-forecasting` | Stages 4 to 7 for time series | Backtesting, lag features, statistical / ML / deep / foundation models, hierarchical reconciliation, forecast metrics |
| `ds-evaluation` | Stage 9 | Metrics by task, threshold and calibration, intervals, slices, error analysis, explainability, sign-off, model card |

Each step skill has a `SKILL.md` (decision table + checks + gate) and a `references/` folder holding
the full method menu, verified against primary sources on the date stated in the file.

## How the pieces fit

1. The orchestrator frames the problem and picks the path (classification, regression,
   forecasting, clustering, anomaly).
2. Each step skill screens its candidate methods under the shared CV and hands back the best
   two or three, not one, with the evidence.
3. `ds-method-comparison` runs the joint search: every carried-forward choice becomes a
   categorical hyperparameter next to the model's own, in a conditional space.
4. Selection uses repeated seeds, a one-standard-error rule, and a locked holdout separate from
   the final test set. The test set is scored once, to report, never to choose.
5. Correctness gates (leakage, split design, sentinel cleanup) sit outside the search. They are
   not tunable.
6. Every stage writes a decision-log block: considered / applied because / skipped because.
   Nothing is used or left out without a reason tied to this data.
7. Compute is not assumed. Candidates are chosen from data size and problem type. When a method
   needs heavy compute (GPU, long search), the skill says so and asks, instead of dropping it.

## Install

Claude Code loads skills from `%USERPROFILE%\.claude\skills\` (user-wide) or
`.claude\skills\` inside a project. Copy each skill folder there, or create a junction so edits in
this repo take effect immediately:

```powershell
$src = "C:\dev\important repo\Data Scientist Skill Set"
$dst = "$env:USERPROFILE\.claude\skills"
Get-ChildItem $src -Directory | Where-Object { Test-Path "$($_.FullName)\SKILL.md" } | ForEach-Object {
  New-Item -ItemType Junction -Path "$dst\$($_.Name)" -Target $_.FullName -Force
}
```

## Evals

`evals/` holds three synthetic test datasets with planted defects, their answer keys and the test
prompts; `evals/README.md` describes them. Test agents are told never to open `evals/` except the
one CSV they are given. Results live in `data-scientist-workspace/iteration-N/`, one folder per
eval with `with_skill/` and `without_skill/` runs and their grading.

**Hygiene rule:** skills must never quote evaluation datasets, their column names or their results.
Iteration 1 did, and the test agent used those quotes as answers. Examples in skills use a neutral
domain (loan default) that matches no test set; `tools/validate_skills.py` plus a grep for test-set
column names should both come back clean before any test run.

## Provenance of facts

Every version, release date, licence and API name in the skills was checked on 2026-09-27 against
PyPI's JSON API, GitHub, package source (downloaded wheels) or local imports, and each reference
file states its source and date. When a library or version looks stale, recheck it the same way.

### Install on macOS or Linux

Claude Code reads `~/.claude/skills/` there. Copy the folders, or symlink them
so edits in a clone take effect immediately:

```bash
git clone https://github.com/apoorva2121/data-scientist-skills.git
cd data-scientist-skills
mkdir -p ~/.claude/skills
for d in */; do [ -f "$d/SKILL.md" ] && ln -sfn "$PWD/${d%/}" ~/.claude/skills/"${d%/}"; done
```

Then say "use the data-scientist skill" in any data project.

## Licence

[MIT](LICENSE). Fork these, adapt them to your own stack, and you need not
credit this repository in your own project's skills.
