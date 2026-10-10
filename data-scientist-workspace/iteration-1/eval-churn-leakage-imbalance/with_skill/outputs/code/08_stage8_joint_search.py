"""
Stage 8 - Joint pipeline search (ds-method-comparison), adapted from
ds-method-comparison/scripts/joint_search.py for this project's carried-forward choices:

  - NUM/CAT: Stage 6's final 8-feature set (region and last_invoice_amount dropped on evidence).
  - outliers slot: none / log1p_skewed(avg_daily_usage_mb) - the two Stage 5 survivors.
  - missing_values slot: native_nan (gbdt) / median (no indicator - Stage 6 dropped
    last_invoice_amount, the only column with informative missingness, so the remaining
    MCAR-like columns (total_charges, age) don't need an indicator).
  - encoding: fixed to one-hot (region, the only column that needed a real encoding choice,
    was dropped at Stage 6; contract_type/payment_method are low-cardinality, one-hot only).
  - Families: linear (logreg, class_weight="balanced" fixed) and gbdt (CatBoost,
    auto_class_weights="Balanced" fixed) - the two Stage 7 survivors. CatBoost's search ranges
    are chosen specifically to address the Stage 7 learning-curve finding (severe overfitting at
    default settings: train PR-AUC 0.94-1.00 vs val 0.13-0.16) via depth/l2/iteration regularization.

Run: python joint_search-adapted, trials=60, timeout=600s (10 min), seeds=3, top=5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-method-comparison/scripts")
sys.path.insert(0, r"C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts")
from cv_folds import load_split  # noqa: E402
from screen import evaluate, summarize, topk_scorer  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = r"C:/dev/important repo/Data Scientist Skill Set/data-scientist-workspace/iteration-1/eval-churn-leakage-imbalance/with_skill/outputs"

# ============================================================ ADAPT: columns
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "avg_daily_usage_mb", "age"]
CAT = ["contract_type", "payment_method"]
SKEWED = ["avg_daily_usage_mb"]


def _log1p_skewed(X):
    X = pd.DataFrame(X, columns=NUM).copy()
    for c in SKEWED:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.to_numpy()


SLOTS = {
    "outliers": {
        "none": lambda fam: "passthrough",
        "log1p_skewed": lambda fam: FunctionTransformer(_log1p_skewed),
    },
    "missing_values": {
        "native_nan": lambda fam: "passthrough",
        "median": lambda fam: SimpleImputer(strategy="median"),
    },
}


# ============================================================ ADAPT: model families
def _linear(trial):
    return LogisticRegression(C=trial.suggest_float("linear_C", 1e-3, 10, log=True),
                               max_iter=3000, class_weight="balanced")


def _gbdt(trial):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=trial.suggest_int("gbdt_iterations", 50, 400, step=50),
        depth=trial.suggest_int("gbdt_depth", 2, 6),
        learning_rate=trial.suggest_float("gbdt_lr", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("gbdt_l2", 1.0, 30.0, log=True),
        auto_class_weights="Balanced", verbose=False, random_state=0, allow_writing_files=False)


FAMILIES = {"linear": _linear, "gbdt": _gbdt}
COMPLEXITY = {"linear": 0, "gbdt": 2}
NEEDS_SCALING = {"linear"}
# native_nan requires no imputation, which only boosted trees can consume; offer it only to gbdt.
MISSING_APPLIES_TO = {"native_nan": ["gbdt"], "median": ["linear", "gbdt"]}
OUTLIER_APPLIES_TO = {"none": ["linear", "gbdt"], "log1p_skewed": ["linear"]}  # trees insensitive, screened once


def options(slot, fam):
    applies = MISSING_APPLIES_TO if slot == "missing_values" else OUTLIER_APPLIES_TO
    return [name for name in SLOTS[slot] if fam in applies[name]]


def build(fam: str, choices: dict, model) -> Pipeline:
    num_steps = [SLOTS["outliers"][choices.get("outliers", "none")](fam),
                 SLOTS["missing_values"][choices.get("missing_values", "native_nan")](fam)]
    if fam in NEEDS_SCALING:
        num_steps.append(StandardScaler())
    num_steps = [st for st in num_steps if st != "passthrough"] or ["passthrough"]
    num_pipe = num_steps[0] if len(num_steps) == 1 else make_pipeline(*num_steps)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("num", num_pipe, NUM), ("cat", enc, CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=OUT + "/cleaned_data.csv")
    ap.add_argument("--split", default=OUT + "/split")
    ap.add_argument("--space", default=OUT + "/search_space.json")
    ap.add_argument("--id", default="customer_id")
    ap.add_argument("--target", default="churned")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=OUT + "/search")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    s = load_split(Path(a.split) / "split.csv", ids=df[a.id])
    space = json.loads(Path(a.space).read_text())
    metric = space["primary_metric"]
    scorer = topk_scorer(float(metric.split("@")[1])) if metric.startswith("precision@") else get_scorer(metric)
    noise = space.get("noise_floor_sd")
    if isinstance(noise, dict):
        noise = noise.get("fold_sd")
    tr = df[s.train_mask].reset_index(drop=True)
    X, y = tr[NUM + CAT], tr[a.target].to_numpy()
    families = list(FAMILIES)

    def objective(trial):
        fam = trial.suggest_categorical("family", families)
        choices = {}
        for slot in SLOTS:
            opts = options(slot, fam)
            if opts:
                choices[slot] = trial.suggest_categorical(f"{slot}|{fam}", opts)
        pipe = build(fam, choices, FAMILIES[fam](trial))
        scores = []
        for k, (i_tr, i_va) in enumerate(s.folds):
            p = clone(pipe).fit(X.iloc[i_tr], y[i_tr])
            scores.append(scorer(p, X.iloc[i_va], y[i_va]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("choices", choices)
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, n_startup_trials=15),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{len(study.trials)} trials ({len(done)} complete, "
          f"{sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)} pruned) in {time.time() - t0:.0f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)

    seen, finalists = set(), {}
    for t in sorted(done, key=lambda t: -t.value):
        key = json.dumps({"family": t.params["family"], **t.user_attrs["choices"]}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        fam = t.params["family"]
        finalists[f"T{t.number}:{fam}:" + ",".join(f"{k}={v}" for k, v in t.user_attrs["choices"].items())] = (t, fam)
        if len(finalists) == a.top:
            break

    def fixed_trial(params):
        return optuna.trial.FixedTrial(params)

    pipes = {name: build(fam, t.user_attrs["choices"], FAMILIES[fam](fixed_trial(t.params)))
             for name, (t, fam) in finalists.items()}
    seed_param = "model__random_state" if all(hasattr(p.named_steps["model"], "random_state") for p in pipes.values()) else None
    long = evaluate(pipes, X, y, folds=s.folds, scoring=scorer, seeds=range(a.seeds), seed_param=seed_param)
    table = summarize(long, noise_floor_sd=noise)
    active = {n: sum(v not in ("none", "native_nan", "passthrough") for v in finalists[n][0].user_attrs["choices"].values())
              for n in finalists}
    table["family"] = [finalists[n][1] for n in table["name"]]
    position = {n: sum(list(SLOTS[slot]).index(v) for slot, v in finalists[n][0].user_attrs["choices"].items())
                for n in finalists}
    table["complexity"] = [(COMPLEXITY.get(finalists[n][1], 9), active[n], position[n]) for n in table["name"]]
    table.to_csv(out / "finalists.csv", index=False)
    print(table.drop(columns="complexity").to_string(index=False))

    best_by_mean = table.iloc[0]["name"]
    pick = table[table["keep"]].sort_values("complexity").iloc[0]["name"]
    print(f"\nbest by mean: {best_by_mean}\n1-SE pick   : {pick}")

    ho = df[s.holdout_mask]
    Xh, yh = ho[NUM + CAT], ho[a.target].to_numpy()
    holdout = {}
    for name in dict.fromkeys([pick, best_by_mean]):
        p = clone(pipes[name]).fit(X, y)
        holdout[name] = float(scorer(p, Xh, yh))
        print(f"holdout {metric}: {holdout[name]:.4f}  <- {name}")
    cv_mean = float(table.set_index("name").loc[pick, "mean"])
    result = {"metric": metric, "pick": pick, "best_by_mean": best_by_mean, "pick_params": finalists[pick][0].params,
              "pick_choices": finalists[pick][0].user_attrs["choices"], "cv_mean_pick": cv_mean, "holdout": holdout,
              "noise_floor_sd": noise, "n_trials": len(study.trials)}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"written: {out / 'result.json'}")


if __name__ == "__main__":
    main()
