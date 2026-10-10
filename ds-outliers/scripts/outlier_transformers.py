"""In-fold outlier treatments as scikit-learn transformers (thresholds learned in `fit` only).

    from outlier_transformers import Winsorizer, RobustClipper, FarValueNuller, AnomalyScore
    num_pipe = make_pipeline(FarValueNuller(factor=10), SimpleImputer(...), Winsorizer(0.01, 0.99))

All accept arrays or DataFrames, keep the column count (except AnomalyScore, which appends one
column), and pass NaN through untouched so a later imputer still sees missing values.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] quantiles learned on the training fold."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class RobustClipper(BaseEstimator, TransformerMixin):
    """Clip at median +/- k robust SDs (robust SD = MAD / 0.6745), learned on the training fold."""

    def __init__(self, k: float = 3.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.med_), axis=0) / 0.6745
        self.mad_ = np.where(mad > 0, mad, np.nanstd(X, axis=0))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.med_ - self.k * self.mad_, self.med_ + self.k * self.mad_)


class FarValueNuller(BaseEstimator, TransformerMixin):
    """Turn values beyond `factor` x the training p99 (or below p01 / factor for positive data) into NaN.

    For suspected entry or unit errors that no domain rule catches exactly. Put an imputer after it.
    """

    def __init__(self, factor: float = 10.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.p99_ = np.nanquantile(X, 0.99, axis=0)
        self.p01_ = np.nanquantile(X, 0.01, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        hi = np.where(self.p99_ > 0, self.factor * self.p99_, np.inf)
        lo = np.where(self.p01_ > 0, self.p01_ / self.factor, -np.inf)
        X[(X > hi) | (X < lo)] = np.nan
        return X


class AnomalyScore(BaseEstimator, TransformerMixin):
    """Append an IsolationForest anomaly score (fitted on the training fold) as an extra feature.

    For when extremes are signal: the model can use "how unusual is this row" directly.
    Expects numeric input without NaN (place it after the imputer).
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.n_estimators, self.random_state = n_estimators, random_state

    def fit(self, X, y=None):
        self.iso_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state).fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([X, -self.iso_.score_samples(X)])
