"""Stage 2: carve and freeze the test set, a locked holdout, and CV folds, in one reproducible step.

Usage:
    python make_split.py DATA.csv --target TARGET --scheme stratified|random|group|time|group_time
        [--id ID_COL or COL1,COL2] [--group GROUP_COL] [--time TIME_COL]
        [--test-size 0.2 | --test-periods N] [--holdout-size 0.15 | --holdout-periods N]
        [--n-splits 5] [--gap-periods 0] [--seed 42] [--out split/]

Writes (no copies of the data, only row ids):
  split/split.csv        id, part (train | holdout | test), fold
  split/split_spec.json  scheme, parameters, sizes, target summary per part, time range per part,
                         group overlap between parts, sha256 of the sorted test ids

Fold semantics
  stratified / random / group : fold k = validation rows of fold k; train on the other train-part rows.
  time / group_time           : expanding window. fold k (0..K-1) = validation block k; train on rows
                                with fold in {-1, 0..k-1} (fold -1 = initial window, never validated).
                                --gap-periods leaves that many periods out between train and validation.
Holdout and test rows have fold = -2. With --drop-overlap-groups (group_time), train rows whose
group reappears in holdout or test get part = excluded and fold = -3; they are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)


def _strata(y: pd.Series) -> np.ndarray:
    """Class labels for classification; quantile bins for a continuous target."""
    if y.nunique() <= 20 or not pd.api.types.is_numeric_dtype(y):
        return y.astype(str).to_numpy()
    return pd.qcut(y.rank(method="first"), q=10, labels=False).to_numpy()


def _periods_split(periods: np.ndarray, n_rows_per_period: pd.Series, frac: float | None, n_periods: int | None):
    """Return the set of most recent periods holding ~frac of rows (or exactly n_periods)."""
    ordered = np.sort(periods)[::-1]
    if n_periods:
        return set(ordered[:n_periods])
    target, acc, chosen = frac * n_rows_per_period.sum(), 0, set()
    for p in ordered:
        if acc >= target:
            break
        chosen.add(p)
        acc += n_rows_per_period[p]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--scheme", required=True, choices=["stratified", "random", "group", "time", "group_time"])
    ap.add_argument("--id", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--time", default="")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--test-periods", type=int, default=0)
    ap.add_argument("--holdout-size", type=float, default=0.15)
    ap.add_argument("--holdout-periods", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap-periods", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-overlap-groups", action="store_true",
                    help="group_time only: exclude train rows whose group also appears in holdout or test "
                         "(deployment on new entities). Excluded rows get part='excluded'.")
    ap.add_argument("--out", default="split")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    if a.id:
        # A composite key such as "entity_id,period" (forecasting panels) is joined with "|".
        id_cols = [c.strip() for c in a.id.split(",")]
        ids = df[id_cols].astype(str).agg("|".join, axis=1) if len(id_cols) > 1 else df[id_cols[0]].astype(str)
        if not ids.is_unique:
            raise SystemExit(f"--id {a.id} is not unique; dedupe at Stage 1 first")
    else:
        ids = pd.Series(np.arange(len(df)).astype(str), name="_row_id")
    y = df[a.target]
    part = np.array(["train"] * len(df), dtype=object)
    fold = np.full(len(df), -2)
    rng = a.seed

    if a.scheme in ("group", "group_time") and not a.group:
        raise SystemExit("--group is required for this scheme")
    if a.scheme in ("time", "group_time") and not a.time:
        raise SystemExit("--time is required for this scheme")

    if a.scheme in ("stratified", "random", "group"):
        idx = np.arange(len(df))
        g = df[a.group].astype(str).to_numpy() if a.group else None
        strata = _strata(y)

        def holdout_split(sub_idx, size, seed):
            if a.scheme == "stratified":
                s = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
                return next(s.split(sub_idx, strata[sub_idx]))
            if a.scheme == "random":
                return next(ShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(sub_idx))
            return next(GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(sub_idx, groups=g[sub_idx]))

        keep, te = holdout_split(idx, a.test_size, rng)
        part[idx[te]] = "test"
        rest = idx[keep]
        if a.holdout_size > 0:
            keep2, ho = holdout_split(rest, a.holdout_size / (1 - a.test_size), rng + 1)
            part[rest[ho]] = "holdout"
            rest = rest[keep2]
        if a.scheme == "stratified":
            cv = StratifiedKFold(n_splits=a.n_splits, shuffle=True, random_state=rng)
            splits = cv.split(rest, strata[rest])
        elif a.scheme == "random":
            splits = KFold(n_splits=a.n_splits, shuffle=True, random_state=rng).split(rest)
        else:
            classification = y.nunique() <= 20
            cv = (StratifiedGroupKFold(n_splits=a.n_splits, shuffle=True, random_state=rng) if classification
                  else GroupKFold(n_splits=a.n_splits))
            splits = cv.split(rest, strata[rest] if classification else None, groups=g[rest])
        for k, (_, va) in enumerate(splits):
            fold[rest[va]] = k
    else:
        t = pd.to_datetime(df[a.time])
        counts = t.value_counts()
        periods = counts.index.to_numpy()
        test_p = _periods_split(periods, counts, a.test_size, a.test_periods)
        part[t.isin(test_p).to_numpy()] = "test"
        remaining = counts.drop(list(test_p))
        if a.holdout_size > 0 or a.holdout_periods:
            frac = a.holdout_size * counts.sum() / remaining.sum() if not a.holdout_periods else None
            ho_p = _periods_split(remaining.index.to_numpy(), remaining, frac, a.holdout_periods)
            part[t.isin(ho_p).to_numpy()] = "holdout"
            remaining = remaining.drop(list(ho_p))
        train_periods = np.sort(remaining.index.to_numpy())
        # expanding window over periods: initial window = first half, then K equal validation blocks
        n = len(train_periods)
        start = n // 2
        blocks = np.array_split(train_periods[start:], a.n_splits)
        pmap = {p: -1 for p in train_periods[:start]}
        for k, b in enumerate(blocks):
            for p in b:
                pmap[p] = k
        is_train = part == "train"
        fold[is_train] = t[is_train].map(pmap).to_numpy()

    n_excluded = 0
    if a.drop_overlap_groups:
        if a.scheme != "group_time":
            raise SystemExit("--drop-overlap-groups applies to --scheme group_time only")
        g = df[a.group].astype(str).to_numpy()
        later = set(g[(part == "holdout") | (part == "test")])
        drop = (part == "train") & np.isin(g, list(later))
        part[drop] = "excluded"
        fold[drop] = -3
        n_excluded = int(drop.sum())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"id": ids, "part": part, "fold": fold})
    if a.time:
        # dense rank of the time column; cv_folds uses it to apply --gap-periods
        res["period"] = pd.to_datetime(df[a.time]).rank(method="dense").astype(int).to_numpy()
    res.to_csv(out / "split.csv", index=False)

    test_ids = sorted(ids[part == "test"])
    spec = {
        "scheme": a.scheme,
        "params": {k: v for k, v in vars(a).items() if k not in ("data", "out")},
        "sizes": res["part"].value_counts().to_dict(),
        "folds": {int(k): int(v) for k, v in res.loc[res.part == "train", "fold"].value_counts().sort_index().items()},
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "excluded_overlap_rows": n_excluded,
        "target_by_part": {},
    }
    for p in ("train", "holdout", "test"):
        yp = y[part == p]
        if len(yp) == 0:
            continue
        spec["target_by_part"][p] = (
            {str(k): round(float(v), 4) for k, v in yp.value_counts(normalize=True).sort_index().items()}
            if y.nunique() <= 20 else {"mean": round(float(yp.mean()), 3), "median": round(float(yp.median()), 3)}
        )
    if a.time:
        t = pd.to_datetime(df[a.time])
        spec["time_range_by_part"] = {p: [str(t[part == p].min())[:10], str(t[part == p].max())[:10]]
                                      for p in ("train", "holdout", "test") if (part == p).any()}
    if a.group:
        g = df[a.group].astype(str)
        sets = {p: set(g[part == p]) for p in ("train", "holdout", "test")}
        spec["group_overlap"] = {"train&test": len(sets["train"] & sets["test"]),
                                 "train&holdout": len(sets["train"] & sets["holdout"]),
                                 "holdout&test": len(sets["holdout"] & sets["test"])}
        if a.scheme == "group":
            tr = res[res.part == "train"]
            leaks = sum(g[tr.index][tr.fold == k].isin(g[tr.index][tr.fold != k]).any() for k in range(a.n_splits))
            spec["group_overlap"]["folds_sharing_groups"] = int(leaks)
    (out / "split_spec.json").write_text(json.dumps(spec, indent=2, default=str))
    print(json.dumps({k: spec[k] for k in spec if k != "params"}, indent=2, default=str))


if __name__ == "__main__":
    main()
