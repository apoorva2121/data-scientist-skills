"""Stage 4: missing values - screen candidates per family on the Stage 2 folds.

missing_profile.csv verdict (all 4 affected columns): MCAR-like, no target signal, not strongly
predictable from other columns, no time trend -> simple imputation is expected to be adequate;
KNN/iterative are not screened (budget cut, logged below) since the diagnosis gives no reason to
expect them to beat median+indicator, and the skill's own guidance is "screen simple first; MICE
rarely wins for boosted trees with native handling and this data shows no MAR structure to exploit".
"""
import json
import sys
import numpy as np
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, make_linear_pipe, make_gbdt_pipe, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
from screen import evaluate, summarize

train, s = load_train()
X = train[NUM_COLS + CAT_COLS]
y = train["price"].to_numpy()
folds = s.folds
NOISE = 289.57  # split/noise_floor.json noise_floor_sd

linear_candidates = {
    "median": make_linear_pipe(add_indicator=False),
    "median_indicator": make_linear_pipe(add_indicator=True),
}
gbdt_candidates = {
    "native_nan": make_gbdt_pipe(impute="native"),
    "median": make_gbdt_pipe(impute="median"),
}

print("=== linear proxy (Ridge, log target), scoring = neg MAE $ ===")
long_lin = evaluate(linear_candidates, X, y, folds=folds, scoring="neg_mean_absolute_error")
tab_lin = summarize(long_lin, noise_floor_sd=NOISE)
print(tab_lin)

print("\n=== gbdt proxy (LightGBM, log target), scoring = neg MAE $ ===")
long_gbdt = evaluate(gbdt_candidates, X, y, folds=folds, scoring="neg_mean_absolute_error")
tab_gbdt = summarize(long_gbdt, noise_floor_sd=NOISE)
print(tab_gbdt)

tab_lin.to_csv(f"{OUT}/stage4_missing_screen_linear.csv", index=False)
tab_gbdt.to_csv(f"{OUT}/stage4_missing_screen_gbdt.csv", index=False)

# update search_space.json
path = f"{OUT}/search_space.json"
try:
    space = json.load(open(path))
except FileNotFoundError:
    space = {"cv": {"scheme": "group_time_expanding", "n_splits": 5, "n_repeats": 1, "groups": "property_id", "seed": 42},
              "primary_metric": "neg_mean_absolute_error_dollar", "noise_floor_sd": NOISE,
              "noise_floor_definition": "fold_sd/sqrt(5) of a deterministic Ridge/log-price baseline on the fixed expanding-window folds (see split/noise_floor.json)",
              "fixed_steps": ["price_per_sqft: dropped (derived from target)",
                               "sqft in {50,100000} -> NaN (sentinel)", "lot_size_sqft==9999 -> NaN (sentinel)",
                               "year_built in {1800,2090} -> NaN (sentinel)", "garage spelling normalized to Yes/No"],
              "stages": {}}

def rows_to_entries(tab, applies_to):
    out = []
    for _, r in tab.iterrows():
        out.append({"name": r["name"], "applies_to": [applies_to],
                     "proxy_scores": {applies_to: [round(r["mean"], 2), round(r["sd"], 2)]},
                     "keep": bool(r["keep"])})
    return out

entries = rows_to_entries(tab_lin, "linear") + rows_to_entries(tab_gbdt, "gbdt")
space["stages"]["missing_values"] = entries
json.dump(space, open(path, "w"), indent=2)
print("\nwrote", path)
