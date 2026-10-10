# Evaluation: metrics, tools and maintenance status

Versions from PyPI's JSON API on 2026-09-27. Class and function names confirmed from package
source (MAPIE 1.5.0, fairlearn 0.14.0, shap 0.52.0) or local imports (scikit-learn 1.7.2).

## Metrics by task

| Task | Primary candidates | Misleading when |
|---|---|---|
| Binary, balanced | ROC-AUC, log loss, accuracy | - |
| Binary, imbalanced | PR-AUC (average precision), precision/recall at capacity, F-beta, MCC | Accuracy and ROC-AUC look good while the minority class is missed |
| Probabilities consumed | Log loss, Brier, ECE | Ranking metrics ignore calibration |
| Multiclass | Macro F1, balanced accuracy, log loss, per-class recall | Micro averages hide small classes |
| Regression | MAE (robust), RMSE (punishes large errors), MAPE only far from zero, R-squared for context | RMSE on heavy tails is dominated by a few rows |
| Quantiles / intervals | Pinball loss, coverage, width | Coverage without width rewards useless wide intervals |
| Ranking | NDCG@k, MAP, MRR | Global metrics ignore the top of the list |
| Clustering | Silhouette, Davies-Bouldin, Calinski-Harabasz; ARI/NMI against labels or across resamples | Internal indices favour convex clusters |
| Forecasting | MASE, WAPE, CRPS, coverage per horizon (see `ds-forecasting`) | MAPE with zeros |

## Comparing two models honestly

| Situation | Method |
|---|---|
| Same CV folds | Paired differences with the Nadeau-Bengio corrected variance (`ds-method-comparison/scripts/screen.py`) |
| Same test rows, any metric | Paired bootstrap of the metric difference (resample rows, recompute both) |
| Two classifiers' error patterns | McNemar test on the disagreement table |
| Two ROC-AUCs on the same rows | DeLong test (no maintained scikit-learn implementation; paired bootstrap is the practical alternative) |
| Many datasets | Bayesian signed-rank (`baycomp` 1.0.3, 2023-07-22, quiet) or Wilcoxon |
| Multiple comparisons | Holm correction; better, a locked holdout |

## Tools

| Tool | pip | Latest (released) | Maintained? | Use |
|---|---|---|---|---|
| MAPIE | `mapie` | 1.5.0 (2026-08-05) | yes | Conformal regression, classification, time series, risk control |
| crepes | `crepes` | 0.9.1 (2026-06-12) | yes | Conformal regressors and predictive systems |
| puncc | `puncc` | 0.9.3 (2026-06-23) | yes | Conformal prediction toolkit |
| netcal | `netcal` | 1.4.0 (2026-04-16) | yes | Temperature, beta, histogram binning, calibration metrics |
| Venn-ABERS | `venn-abers` | 1.5.4 (2026-08-12) | yes | Calibrated probability intervals |
| fairlearn | `fairlearn` | 0.14.0 (2026-06-07) | yes | `MetricFrame`, disparity metrics, `ThresholdOptimizer` |
| Aequitas | `aequitas` | 1.1.0 (2026-02-03) | yes | Bias audit reports |
| AIF360 | `aif360` | 0.6.1 (2024-04-08) | quiet | Broad fairness metrics and mitigations |
| SHAP | `shap` | 0.52.0 (2026-05-28) | yes | `TreeExplainer`, `PermutationExplainer`, `KernelExplainer` |
| shapiq | `shapiq` | 1.7.0 (2026-08-27) | yes | Shapley interactions, `TabularExplainer` |
| DiCE | `dice-ml` | 0.12 (2025-07-13) | yes | Counterfactual explanations |
| Alibi | `alibi` | 0.9.6 (2024-04-18) | quiet | ALE, anchors, counterfactuals |
| LIME | `lime` | 0.2.0.1 (2020-06-26) | no | Superseded; prefer SHAP |
| Evidently | `evidently` | 0.7.23 (2026-09-11) | yes | Drift and quality reports, test suites |
| NannyML | `nannyml` | 0.13.1 (2025-07-12) | yes | Label-free performance estimation (CBPE, DLE), drift |
| Alibi Detect | `alibi-detect` | 0.13.0 (2025-12-11) | yes | Statistical drift and outlier detectors |
| skops | `skops` | 0.16.0 (2026-09-24) | yes | Secure scikit-learn persistence (`skops.io`) |
| skl2onnx | `skl2onnx` | 1.20.0 (2026-01-30) | yes | Export pipelines to ONNX |
| MLflow | `mlflow` | 3.16.1 (2026-09-16) | yes | Experiment tracking, model registry |
| Weights & Biases | `wandb` | 0.30.0 (2026-09-09) | yes | Experiment tracking |
| DVC | `dvc` | 3.67.1 (2026-03-31) | yes | Data and pipeline versioning |
| pandera | `pandera` | 0.33.1 (2026-09-01) | yes | Batch data validation from Stage 1 rules |

## Drift statistics

| Statistic | Use | Rule of thumb |
|---|---|---|
| PSI (population stability index) | Binned feature or score distributions | Under 0.1 stable, 0.1-0.25 moderate, over 0.25 large (industry convention, not a law) |
| Kolmogorov-Smirnov | Continuous features | Very sensitive at large n; look at effect size too |
| Jensen-Shannon / Wasserstein | Any distribution | Bounded (JS) or in feature units (Wasserstein) |
| Chi-square | Categorical features | Sensitive at large n |
| Adversarial validation | Whole-row shift | AUC near 0.5 means no detectable shift |
