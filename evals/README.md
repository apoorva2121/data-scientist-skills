# Evaluation datasets (do not show to agents under test)

`make_datasets.py` generates three synthetic datasets with planted defects and writes an answer key
for each under `answer_keys/`. `evals.json` holds the prompts. Grading assertions live in each
eval's `eval_metadata.json` under `data-scientist-workspace/iteration-N/`.

- `churn.csv`: binary classification with two leak columns, sentinels, dirty categories,
  impossible ages, informative missingness, entry errors in usage, duplicates, 9% positives,
  and a 10% call-capacity constraint.
- `house_prices.csv`: regression with a derived-from-target leak, repeat sales (group leakage),
  price drift over time, a skewed target, MCAR missingness, inconsistent codings, impossible values.
- `store_sales.csv`: 20 weekly series with a same-week leak, sentinel sales, duplicate keys,
  gaps, a level shift, an intermittent series, and known-future promo and holiday covariates.

Regenerate with `python evals/make_datasets.py` (seeded, deterministic).

## Change log

- 2026-09-27: fixed three answer-key errors found by the first baseline runs (tax value leaked sale
  noise; invoice missingness depended on the label; inventory at the forecast origin is legitimate),
  kept resales inside the data window, strengthened the invoice missingness signal.
