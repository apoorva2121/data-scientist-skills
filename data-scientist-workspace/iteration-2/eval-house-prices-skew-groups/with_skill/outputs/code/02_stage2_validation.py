"""Stage 2: validation design. group_time split (property_id x sold_date) + group-overlap fix
+ noise floor, per ds-validation-design.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs"
sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\ds-validation-design\scripts")
from cv_folds import load_split, verify_test_untouched

df = pd.read_csv(f"{OUT}/house_prices_stage1_clean.csv", parse_dates=["sold_date"])
df["listing_id"] = df["listing_id"].astype(str)
df["property_id"] = df["property_id"].astype(str)

split = pd.read_csv(f"{OUT}/split/split.csv", dtype={"id": str})
split = split.set_index("id").loc[df["listing_id"]].reset_index()

# --- deployment decision: pricing tool for NEW listings -> treat test/holdout as unseen entities.
# Drop train rows whose property_id also appears in holdout or test (conservative: a resold
# property's earlier sale would otherwise leak its static attributes, e.g. tax_assessed_value,
# lot_size, into the exact row being scored).
prop = df.set_index("listing_id")["property_id"]
holdout_props = set(prop[split.set_index("id")["part"].eq("holdout").reindex(prop.index).fillna(False).to_numpy()])
test_props = set(prop[split.set_index("id")["part"].eq("test").reindex(prop.index).fillna(False).to_numpy()])
# simpler/robust: join part onto listing_id directly
sp = split.set_index("id")
sp["property_id"] = prop
holdout_props = set(sp.loc[sp["part"] == "holdout", "property_id"])
test_props = set(sp.loc[sp["part"] == "test", "property_id"])
drop_mask = (sp["part"] == "train") & sp["property_id"].isin(holdout_props | test_props)
n_dropped = int(drop_mask.sum())
sp.loc[drop_mask, "part"] = "excluded_group_overlap"

sp.reset_index().drop(columns=["property_id"]).to_csv(f"{OUT}/split/split.csv", index=False)

print(f"train rows dropped for property_id overlap with holdout/test: {n_dropped}")
print(sp["part"].value_counts())

# reload the cleaned split for CV/noise-floor work
s = load_split(f"{OUT}/split/split.csv", ids=df["listing_id"])
print("verify_test_untouched:", verify_test_untouched(f"{OUT}/split"))
print("n folds:", len(s.folds))
for k, (tr, va) in enumerate(s.folds):
    print(f"  fold {k}: train={len(tr)} val={len(va)}")

train_df = df[s.train_mask].reset_index(drop=True)
holdout_df = df[s.holdout_mask].reset_index(drop=True)
test_df = df[s.test_mask].reset_index(drop=True)

# --- noise floor: one simple pipeline (Ridge, log target), deterministic -> fold_sd / sqrt(k)
num_cols = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft", "distance_to_center_km", "tax_assessed_value"]
cat_cols = ["neighborhood", "condition", "garage"]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])

X = train_df[num_cols + cat_cols]
y = np.log(train_df["price"])
y_dollar = train_df["price"].to_numpy()

fold_maes = []
for tr, va in s.folds:
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred_log = pipe.predict(X.iloc[va])
    pred = np.exp(pred_log)
    fold_maes.append(mean_absolute_error(y_dollar[va], pred))

fold_maes = np.array(fold_maes)
fold_sd = fold_maes.std(ddof=1)
noise_floor_sd = fold_sd / np.sqrt(len(fold_maes))
print("\nnoise-floor pipeline (Ridge, log target) per-fold MAE ($):", np.round(fold_maes, 1))
print(f"fold_sd={fold_sd:.2f}  noise_floor_sd (fold_sd/sqrt(k))={noise_floor_sd:.2f}  mean_MAE={fold_maes.mean():.1f}")

with open(f"{OUT}/split/noise_floor.json", "w") as f:
    json.dump({
        "pipeline": "Ridge(alpha=1.0) on log(price), median/most_frequent impute, standardize, one-hot",
        "fold_maes_dollar": fold_maes.round(2).tolist(),
        "fold_sd": round(float(fold_sd), 2),
        "noise_floor_sd": round(float(noise_floor_sd), 2),
        "mean_mae_dollar": round(float(fold_maes.mean()), 2),
        "n_folds": len(fold_maes),
        "n_train_rows_after_overlap_fix": len(train_df),
        "n_dropped_for_group_overlap": n_dropped,
    }, f, indent=2)

print("\ntrain/holdout/test sizes:", len(train_df), len(holdout_df), len(test_df))
print("test date range:", test_df["sold_date"].min(), test_df["sold_date"].max())
