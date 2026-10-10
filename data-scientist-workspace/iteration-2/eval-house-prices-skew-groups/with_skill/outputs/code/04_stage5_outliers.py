"""Stage 5: outliers - verdicts from outlier_profile.csv, then screen treatments (linear only;
GBDT is invariant to monotonic clipping of feature values, so "none" is the fixed reference and
is not screened against alternatives - a mechanics-based reason, not a skipped step).
"""
import json
import sys
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from common import load_train, make_linear_pipe, make_gbdt_pipe, NUM_COLS, CAT_COLS, OUT
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-method-comparison\scripts")
from screen import evaluate, summarize

train, s = load_train()
X = train[NUM_COLS + CAT_COLS]
y = train["price"].to_numpy()
folds = s.folds
NOISE = 289.57

linear_candidates = {
    "none": make_linear_pipe(add_indicator=False, outlier="none"),
    "winsorize_p01_p99_tailcols": make_linear_pipe(add_indicator=False, outlier="winsorize"),
}
print("=== linear proxy: outlier treatment (lot_size_sqft, distance_to_center_km) ===")
long_lin = evaluate(linear_candidates, X, y, folds=folds, scoring="neg_mean_absolute_error")
tab_lin = summarize(long_lin, noise_floor_sd=NOISE)
print(tab_lin)
tab_lin.to_csv(f"{OUT}/stage5_outlier_screen_linear.csv", index=False)

# gbdt reference point only (not screened against alternatives - see docstring)
gbdt_ref = evaluate({"none": make_gbdt_pipe(impute="median")}, X, y, folds=folds, scoring="neg_mean_absolute_error")
print("\n=== gbdt reference (none, not screened) ===")
print(summarize(gbdt_ref))

path = f"{OUT}/search_space.json"
space = json.load(open(path))
entries = []
for _, r in tab_lin.iterrows():
    entries.append({"name": r["name"], "applies_to": ["linear"],
                     "proxy_scores": {"linear": [round(r["mean"], 2), round(r["sd"], 2)]},
                     "keep": bool(r["keep"])})
entries.append({"name": "none", "applies_to": ["gbdt"], "proxy_scores": {},
                 "keep": True, "reason": "fixed reference; trees invariant to monotonic clipping, not screened"})
space["stages"]["outliers"] = entries
json.dump(space, open(path, "w"), indent=2)
print("wrote", path)
