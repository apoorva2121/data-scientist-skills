# Model card template

Fill every section; write "not assessed" rather than deleting one. Based on the structure of
Mitchell et al. (2019), "Model Cards for Model Reporting".

```markdown
# Model card: <model name> v<version>
Date: <YYYY-MM-DD>   Owner: <team/person>   Contact: <channel>

## Intended use
- Decision supported: <who acts on the output, and how>
- In scope: <population, period, geography>
- Out of scope: <uses that were not validated>

## Model
- Pipeline: <preprocessing steps> -> <model family> (hyperparameters in <file>)
- Training data: <source, period, rows, positives>; test set: <period, rows>, untouched until <date>
- Library versions: <pinned list or lock file>

## Performance (untouched test set, 95% bootstrap intervals)
| Metric | Model | Business baseline | Simple baseline |
|---|---|---|---|
| <primary> | x (lo, hi) | | |
| <secondary> | | | |
Threshold: <value and how chosen>. Calibration: ECE <value>, method <...>.
Intervals: nominal <80%>, empirical coverage <...>.

## Performance by segment
<slice table; worst slice named with its interval>

## Fairness
<group metrics for protected attributes, or why not applicable>

## Explanations
<top features with direction; known interactions>

## Limitations and risks
<ASSUMED lines from the problem statement; data gaps; drift risks; known failure modes>

## Monitoring and maintenance
- Drift checks: <features, statistic, threshold, cadence>
- Performance tracking: <label delay, metric, alert level>
- Retraining trigger: <observable condition>
- Data validation: <schema file>
```
