"""Stage 10: handoff. Fit the production pipeline on RAW rows (train part only, per the Stage 9
design note), save with skops, reload, and run the inference-readiness checks on raw data."""
import sys
import time
import numpy as np
import pandas as pd
import skops.io as sio

sys.path.insert(0, r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs\code")
from production_pipeline import build_production_pipeline, predict_with_interval, FEATS

OUT = r"C:\dev\important repo\Data Scientist Skill Set\data-scientist-workspace\iteration-2\eval-house-prices-skew-groups\with_skill\outputs"
RAW_PATH = r"C:\dev\important repo\Data Scientist Skill Set\evals\datasets\house_prices.csv"

raw = pd.read_csv(RAW_PATH)
raw["listing_id"] = raw["listing_id"].astype(str)
split = pd.read_csv(f"{OUT}/split/split.csv", dtype={"id": str}).set_index("id")
train_ids = split.index[split["part"] == "train"]
raw_train = raw[raw["listing_id"].isin(train_ids)].reset_index(drop=True)
print("raw training rows:", len(raw_train))

pipe = build_production_pipeline()
pipe.fit(raw_train, raw_train["price"].to_numpy())

path = f"{OUT}/production_pipeline.skops"
sio.dump(pipe, path)
print("saved:", path)

# reload and verify (trusted source: this session's own file)
untrusted = sio.get_untrusted_types(file=path)
print("untrusted types on reload:", untrusted)
pipe2 = sio.load(path, trusted=untrusted)

# inference-readiness: raw uncleaned rows end to end, timing, duplicate determinism
sample = raw.sample(10, random_state=1).reset_index(drop=True)
t0 = time.time()
pred, lower, upper = predict_with_interval(pipe2, sample)
elapsed_ms = (time.time() - t0) * 1000 / len(sample)
print(f"scored {len(sample)} raw rows OK, {elapsed_ms:.2f} ms/row avg")
for i in range(len(sample)):
    print(f"  {sample.loc[i,'listing_id']}: point=${pred[i]:,.0f}  90% interval=[${lower[i]:,.0f}, ${upper[i]:,.0f}]  actual=${sample.loc[i,'price']:,.0f}")

# feed one row twice -> identical
twice = pd.concat([sample.iloc[[0]], sample.iloc[[0]]], ignore_index=True)
p2, _, _ = predict_with_interval(pipe2, twice)
print("identical-row determinism:", "PASS" if p2[0] == p2[1] else "FAIL", p2)

# row with every feature missing except required raw columns present as NaN
blank = sample.iloc[[0]].copy()
for c in ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_sqft", "distance_to_center_km",
          "tax_assessed_value", "neighborhood", "condition", "garage"]:
    blank[c] = np.nan
try:
    p_blank, _, _ = predict_with_interval(pipe2, blank)
    print("all-features-missing row: PASS ->", p_blank)
except Exception as e:
    print("all-features-missing row: FAIL ->", e)

print("\nStage 10 handoff checks complete.")
