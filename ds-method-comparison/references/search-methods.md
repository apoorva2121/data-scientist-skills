# Search methods and tools

Versions and release dates from PyPI's JSON API on 2026-09-27. Sampler and pruner names confirmed by
importing the installed Optuna 5.0.0 the same day; scikit-learn behaviour confirmed on 1.7.2.

## Search strategies

| Strategy | Where | Use when | Caveats |
|---|---|---|---|
| Grid search | `sklearn.model_selection.GridSearchCV` | Two or three discrete settings, cheap fits | Cost multiplies per dimension; wastes trials on unimportant parameters |
| Random search | `RandomizedSearchCV` | Baseline for any search; many dimensions | Needs more trials than model-based search to find narrow optima |
| Successive halving | `HalvingRandomSearchCV` (needs `from sklearn.experimental import enable_halving_search_cv` in 1.7.2) | Many candidates, fits that scale with data or iterations | Early rounds on small budgets can drop late bloomers |
| TPE (Bayesian, tree-structured) | `optuna.samplers.TPESampler(multivariate=True, group=True)` | Default for mixed and conditional spaces, as in `joint_search.py` | Needs 10-20 start-up trials before it beats random |
| Gaussian process | `optuna.samplers.GPSampler` | Few continuous parameters, expensive trials | Weak on categorical and conditional spaces |
| CMA-ES | `optuna.samplers.CmaEsSampler` | Continuous parameters, large trial budgets | Needs the `cmaes` package; no categorical support |
| Quasi-random | `optuna.samplers.QMCSampler` | Better-spread start-up trials | No learning from results |
| Multi-objective | `optuna.samplers.NSGAIISampler` | Trade accuracy against latency, size or fairness | Returns a Pareto front; the choice between front points is a business decision |

## Early stopping of weak trials (pruners, Optuna 5.0.0)

| Pruner | Use when |
|---|---|
| `MedianPruner` | Default; report the running mean after each fold, as `joint_search.py` does |
| `SuccessiveHalvingPruner`, `HyperbandPruner` | Iterative models (boosting rounds, epochs) reporting per iteration |
| `PatientPruner` | Wrap another pruner to tolerate noisy early steps |
| `WilcoxonPruner` | Per-fold or per-instance scores; prunes on a paired test against the best trial |

## Tools

| Tool | pip name | Latest (released) | Use when |
|---|---|---|---|
| Optuna | `optuna` | 5.0.0 (2026-09-07) | Default here: conditional spaces, pruning, study storage, visualisation |
| Hyperopt | `hyperopt` | 0.3.0 (2026-07-24) | Existing Hyperopt code; TPE with nested spaces |
| SMAC3 | `smac` | 2.4.1 (2026-09-10) | Random-forest-based Bayesian optimisation, strong on conditional spaces; multi-fidelity |
| Ray Tune | `ray` (`ray[tune]`) | 2.58.0 (2026-08-23) | Distributing trials over many machines; population-based training |
| AutoGluon (tabular) | `autogluon` | 1.6.3 (2026-09-18) | Ceiling check: multi-layer stacked ensembles of many families with little setup |
| FLAML | `flaml` | 2.7.0 (2026-09-18) | Cheap, fast AutoML ceiling check under a tight time budget |

AutoML frameworks use their own internal validation. For a fair ceiling check, give them only the
train part, then score their final model with the same folds or on the locked holdout; never on test.

## Selection rules

- **Paired comparison on shared folds.** Differences per fold cancel fold difficulty.
- **Nadeau-Bengio corrected variance.** Var(diff) x (1/n + n_val/n_train) instead of Var(diff)/n,
  because CV training sets overlap. Implemented in `scripts/screen.py`.
- **One-standard-error rule.** Among options within one SE of the best, choose the simplest.
- **Distinct finalists.** Compare different designs, each at its best-tuned trial, not hyperparameter
  neighbours of one design.
- **Locked holdout.** Carved at Stage 2, used only to confirm the search result.
- **Multiple comparisons.** The more configurations searched, the more optimistic the best CV score.
  The holdout and the untouched test set are the correction.

## Ensembling methods

| Method | How | Accept when |
|---|---|---|
| Simple average | Mean of finalists' predicted probabilities or values | Gain over the best single pipeline exceeds the noise floor |
| Weighted average | Weights fitted on out-of-fold predictions (non-negative, sum to 1) | As above, and weights are stable across folds |
| Stacking | Linear meta-learner on out-of-fold predictions (`StackingClassifier/Regressor` with `cv=`) | As above; meta-learner never sees in-fold predictions |
| Seed averaging | Same pipeline, several seeds | Stochastic models with visible seed variance |
| Multi-layer stacking | AutoGluon-style layers | Large data and budget; interpretability not required |
