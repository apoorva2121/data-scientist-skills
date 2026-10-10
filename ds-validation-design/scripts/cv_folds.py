"""Load the Stage 2 split so every later stage screens methods on exactly the same folds.

    from cv_folds import load_split
    s = load_split("split/split.csv", ids=df["entity_id"])
    train_df = df[s.train_mask]            # CV rows (excludes holdout and test)
    for tr, va in s.folds:                  # positional indices into train_df
        ...
    holdout_df = df[s.holdout_mask]         # Stage 8 only
    test_df = df[s.test_mask]               # Stage 9 only, once

Fold semantics follow split_spec.json: for "time" and "group_time" schemes, fold k trains on
rows with fold in {-1, 0..k-1} and validates on fold k (expanding window), leaving out the last
`gap_periods` periods before the validation block (set with make_split.py --gap-periods). For the other schemes,
fold k validates on fold k and trains on every other train-part row.
The result is also accepted directly as `cv=` by scikit-learn (an iterable of index pairs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def make_ids(df: pd.DataFrame, id_spec: str) -> pd.Series:
    """Row ids exactly as make_split.py built them: one column, or several joined with '|'.

        s = load_split("split/split.csv", ids=make_ids(df, "entity_id,period"))
    """
    cols = [c.strip() for c in id_spec.split(",")]
    return df[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)


@dataclass
class Split:
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    test_mask: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    scheme: str

    def __iter__(self):
        return iter(self.folds)

    def __len__(self):
        return len(self.folds)


def load_split(split_csv: str | Path, ids: pd.Series | None = None) -> Split:
    split_csv = Path(split_csv)
    spec = json.loads((split_csv.parent / "split_spec.json").read_text())
    sp = pd.read_csv(split_csv, dtype={"id": str})
    if ids is not None:
        sp = sp.set_index("id").loc[pd.Series(ids).astype(str).to_numpy()].reset_index()
    train_mask = (sp["part"] == "train").to_numpy()
    fold = sp.loc[train_mask, "fold"].to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    ks = sorted(k for k in np.unique(fold) if k >= 0)
    expanding = spec["scheme"] in ("time", "group_time")
    gap = int(spec["params"].get("gap_periods", 0) or 0)
    if expanding and "period" not in sp.columns:
        raise ValueError(f"{split_csv} has no 'period' column; regenerate it with the current make_split.py")
    period = sp.loc[train_mask, "period"].to_numpy() if expanding else None
    for k in ks:
        va = np.flatnonzero(fold == k)
        if expanding:
            cutoff = period[va].min() - gap  # train only on periods strictly before the gap
            tr = np.flatnonzero((fold < k) & (period < cutoff))
        else:
            tr = np.flatnonzero(fold != k)
        folds.append((tr, va))
    return Split(train_mask, (sp["part"] == "holdout").to_numpy(), (sp["part"] == "test").to_numpy(),
                 folds, spec["scheme"])


def verify_test_untouched(split_dir: str | Path) -> bool:
    """Recompute the test-id hash; False means the test set changed since Stage 2."""
    import hashlib
    split_dir = Path(split_dir)
    spec = json.loads((split_dir / "split_spec.json").read_text())
    sp = pd.read_csv(split_dir / "split.csv", dtype={"id": str})
    ids = sorted(sp.loc[sp["part"] == "test", "id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest() == spec["test_ids_sha256"]
