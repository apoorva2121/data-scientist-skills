# Cleaning log — Stage 1 (stateless, row-wise only)

- raw rows read: 6150
- drop exact duplicate rows: 150 removed -> 6000 remain
- total_charges: -999 sentinel -> NaN, 360 values affected; total missing now 600 (10.00%)
- age: impossible values {0, 250} -> NaN, 60 values affected (1.00% missing now)
- contract_type: 12 raw spellings -> canonicalized to ['month-to-month', 'one year', 'two year'] (map: strip + lowercase)
- sanity: churn rate overall=0.0865, churn rate among dropped duplicate rows=0.0800 (not concentrated in either class)
- final cleaned rows: 6000, columns: 15
- target class balance after cleaning: {0: 0.9133333333333333, 1: 0.08666666666666667}

Candidate keys / groups / time columns handed to Stage 2:
- key: customer_id (unique after dedup)
- group column: none identified (customer_id unique per row post-dedup, no repeated-entity structure)
- time column: signup_date present but inconsistent with tenure_months (r=0.019) — not usable to build a time-ordered split; no reliable snapshot/extraction date column exists
