"""A business rule as a scikit-learn classifier, so it runs through the same CV, scorers and helpers.

    from rule_baseline import RuleBaseline
    rule = RuleBaseline(lambda X: 2 * (X["plan"] == "monthly") + X["complaints_90d"] / 10)
    long = evaluate({"business_rule": rule, "logreg": pipe}, X, y, folds, scoring="average_precision")

The rule returns a risk score per row (higher = more likely positive). Nothing is fitted, so the
rule sees no labels. Scores are min-max scaled to [0, 1] per call for predict_proba; ranking
metrics (ROC-AUC, PR-AUC, precision at k) are unaffected by that scaling, but do not read the
output as calibrated probabilities.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBaseline(ClassifierMixin, BaseEstimator):
    def __init__(self, rule: Callable | None = None):
        self.rule = rule

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        s = np.asarray(self.rule(X), dtype=float)
        p = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
