# Outliers and anomaly detection: method menu

Versions from PyPI's JSON API on 2026-09-27. PyOD detector names confirmed in the PyOD 3.6.6 PyPI
description the same day. scikit-learn classes confirmed on the installed 1.7.2 (latest release is
1.9.1, 2026-09-10; recheck on upgrade).

## Univariate detection

| Method | Rule | Use when | Caveats |
|---|---|---|---|
| IQR fences (Tukey) | Beyond Q1 - k IQR or Q3 + k IQR, k = 1.5 (mild) or 3 (far) | Default first look | Skewed data flags the whole long tail |
| Z-score | abs((x - mean) / sd) > 3 | Roughly normal columns | Mean and sd are themselves pulled by outliers |
| Robust z (MAD) | 0.6745 abs(x - median) / MAD > 3.5 | Default robust rule (Iglewicz and Hoaglin) | MAD = 0 for columns with a dominant value |
| Percentile | Below p0.5 or above p99.5 | Always flags a fixed share | Flags something even when nothing is wrong |
| Hampel filter | Rolling median +/- k MAD | Time series spikes | Window choice matters |
| Domain rules | Impossible values and combinations | Always, first | Needs the data dictionary or an owner |

## Multivariate detection (scikit-learn 1.7.2)

| Method | Class | Use when | Needs scaling? |
|---|---|---|---|
| Isolation Forest | `sklearn.ensemble.IsolationForest` | General default; mixed shapes, many columns | No |
| Local Outlier Factor | `sklearn.neighbors.LocalOutlierFactor` (`novelty=True` to score new data) | Local density differences | Yes |
| One-class SVM | `sklearn.svm.OneClassSVM`; for large data `sklearn.linear_model.SGDOneClassSVM` | Smooth boundaries, moderate size | Yes |
| Elliptic envelope (robust covariance, MCD) | `sklearn.covariance.EllipticEnvelope` | Roughly Gaussian, few columns | Yes |
| Mahalanobis distance | robust covariance via `MinCovDet` | Correlated numeric columns | Implicit |
| DBSCAN / HDBSCAN noise points | `sklearn.cluster.DBSCAN`, `HDBSCAN` | Density-based structure; rows labelled -1 | Yes |
| Model residuals | Large absolute residual from a simple model | Rows unusual given their features | Depends on model |

## PyOD 3 (pip `pyod`, 3.6.6, released 2026-09-17)

One `fit` / `decision_function` API; import as `from pyod.models.<module> import <Class>`.
Families (names as listed in the PyOD description):

| Family | Detectors |
|---|---|
| Probabilistic and statistical | ECOD, COPOD, HBOS, KDE, GMM, MCD, SOS, QMCD, Sampling |
| Proximity | KNN, LOF, COF, CBLOF, ABOD, INNE, LMDD, ROD |
| Linear | PCA, KPCA, OCSVM, CD |
| Ensembles | IForest, LODA, LSCP, SUOD, DIF, XGBOD (semi-supervised) |
| Neural | AutoEncoder, VAE, DeepSVDD, SO_GAAL, AnoGAN, ALAD, LUNAR, DevNet (semi-supervised) |

Practical defaults: ECOD and COPOD are parameter-free and fast, good first choices on many columns;
IForest is a robust general default; KNN and LOF suit local-density anomalies; the neural detectors
need more data and compute. PyOD 3 also provides `ADEngine` (`pyod.utils.ad_engine`), an MCP server
(`pyod[mcp]`), and a Claude Code skill (`od-expert`).

**Benchmark:** ADBench (Han et al., NeurIPS 2022; package `adbench` 0.1.11, 2023-08-02) compared
unsupervised, semi-supervised and supervised detectors across many datasets. Its broad lessons:
no unsupervised detector wins everywhere, so choose by data shape and combine several; and a small
number of labels, when available, improves detection substantially.

## Treatments

| Treatment | Implementation | Families | Notes |
|---|---|---|---|
| None | - | trees | Default reference for tree-based models |
| Null then impute | `FarValueNuller` or a domain rule, then an imputer | all | For errors |
| Winsorize (cap) | `Winsorizer(0.01, 0.99)` in-fold | linear, distance, neural | Keeps rows, limits leverage |
| Robust clip | `RobustClipper(k=3.5)` in-fold | linear, distance | Symmetric around the median |
| Transform | log1p, Box-Cox, Yeo-Johnson (`PowerTransformer`), `QuantileTransformer` | linear, distance, neural | Stage 6; changes the relationship's shape too |
| Robust scaling | `RobustScaler` | distance, linear | Scales by IQR instead of sd |
| Robust loss | Huber, quantile, absolute error | regression, any family that supports it | For target extremes |
| Anomaly score feature | `AnomalyScore` (Isolation Forest, in-fold) | all | When extremes are signal |
| Remove rows | Confirmed errors only, training rows only | all | Report count and concentration |

`feature-engine` (1.9.4, 2026-02-27) also provides `Winsorizer` and `OutlierTrimmer` transformers
with Gaussian, IQR, MAD and quantile rules.
