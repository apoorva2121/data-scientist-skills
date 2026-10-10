"""Screen candidate pipelines on fixed folds and decide survivors with paired, overlap-corrected tests.

    from screen import evaluate, summarize, topk_scorer
    long = evaluate({"median": pipe_a, "knn": pipe_b}, X, y, folds=s.folds, scoring="average_precision")
    table = summarize(long, noise_floor_sd=0.018)        # one row per candidate, sorted best first
    print(table)                                           # keep rows where keep == True

Why paired: every candidate is scored on the same folds, so fold difficulty cancels out when you
compare per-fold differences. Why corrected: CV training sets overlap, so the naive standard error
of the mean difference is too small. The Nadeau-Bengio correction multiplies the variance of the
differences by (1/n + n_val/n_train) instead of 1/n.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer, make_scorer


def fresh_copy(estimator):
    """Unfitted copy of an estimator or pipeline.

    sklearn.clone() fails for some third-party estimators whose constructor does not round-trip a
    parameter (for example CatBoost with cat_features=[...] raises RuntimeError in scikit-learn
    1.7.2). Fall back to a deep copy of the unfitted object, which is equivalent before fitting.
    """
    import copy
    try:
        return clone(estimator)
    except RuntimeError:
        return copy.deepcopy(estimator)


def topk_scorer(frac: float = 0.10, what: str = "precision") -> Callable:
    """Precision or recall among the top `frac` of rows by score (capacity-limited actions)."""

    def _metric(y_true, y_score):
        y_true = np.asarray(y_true)
        k = max(1, int(round(frac * len(y_true))))
        top = np.argsort(-np.asarray(y_score))[:k]
        hits = y_true[top].sum()
        return hits / k if what == "precision" else hits / max(y_true.sum(), 1)

    return make_scorer(_metric, response_method="predict_proba")


def evaluate(pipes: dict, X: pd.DataFrame, y, folds: Iterable, scoring="roc_auc", seeds: Iterable[int] = (0,),
             seed_param: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fit every pipeline on every fold (and every seed). Returns one row per (name, seed, fold).

    seed_param: name of the pipeline parameter that takes a random seed, e.g. "model__random_state".
    """
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    y = np.asarray(y)
    folds = list(folds)
    rows = []
    for name, pipe in pipes.items():
        t0 = time.time()
        for seed in seeds:
            for f, (tr, va) in enumerate(folds):
                p = fresh_copy(pipe)
                if seed_param:
                    p.set_params(**{seed_param: seed})
                p.fit(X.iloc[tr], y[tr])
                rows.append({"name": name, "seed": seed, "fold": f, "score": scorer(p, X.iloc[va], y[va]),
                             "n_train": len(tr), "n_val": len(va)})
        if verbose:
            s = pd.DataFrame([r for r in rows if r["name"] == name])["score"]
            print(f"{name:32s} {s.mean():.4f} +/- {s.std(ddof=1):.4f}  ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame, higher_is_better: bool = True, noise_floor_sd: float | None = None) -> pd.DataFrame:
    """One row per candidate: mean, sd, paired difference to the best, corrected SE, keep flag.

    keep = not worse than the best by more than one corrected standard error of the PAIRED difference.
    Pairing on shared folds cancels fold difficulty, so the paired SE is the right yardstick here.
    The Stage 2 noise floor (sd of a CV mean) is for numbers NOT measured on the same folds, e.g. a
    CV mean against a holdout score; if given, it is reported as `within_noise_floor` for reference
    and does not change `keep`. Ties are kept; the simplest kept candidate is preferred.
    """
    sign = 1 if higher_is_better else -1
    # Average seeds within each fold first: seeds reduce model randomness, but only folds are
    # independent evidence. Counting seed repeats as pairs would shrink the SE artificially
    # (a deterministic model repeats the identical score for every seed).
    wide = long.groupby(["fold", "name"])["score"].mean().unstack()
    ratio = float((long["n_val"] / long["n_train"]).mean())
    means = wide.mean()
    best = (sign * means).idxmax()
    out = []
    for name in wide.columns:
        d = sign * (wide[name] - wide[best])
        n = d.notna().sum()
        var = d.var(ddof=1) if n > 1 else 0.0
        se = float(np.sqrt((1 / n + ratio) * var)) if n > 1 else float("nan")
        diff = float(d.mean())
        row = {"name": name, "mean": round(float(means[name]), 4), "sd": round(float(wide[name].std(ddof=1)), 4),
               "diff_vs_best": round(diff, 4), "se_corrected": round(se, 4) if np.isfinite(se) else np.nan,
               "n_pairs": int(n), "keep": bool(name == best or (np.isfinite(se) and diff >= -se))}
        if noise_floor_sd is not None:
            # Informational only: would this gap be visible between two separately measured numbers?
            row["within_noise_floor"] = bool(abs(diff) <= noise_floor_sd)
        out.append(row)
    if len(wide) < 10:
        print(f"WARNING: only {len(wide)} folds to pair on; the paired SE is itself noisy. Use repeated CV "
              "(>= 10 folds) where the scheme allows, or treat keep decisions as provisional.")
    return pd.DataFrame(out).sort_values("mean", ascending=not higher_is_better).reset_index(drop=True)
