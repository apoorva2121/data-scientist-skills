# Cleaning log (Stage 1)

1. Loaded raw: 6150 rows, 15 columns.
2. total_charges: -999 sentinel -> null (367 rows affected).
3. age: 0 and 250 (impossible ages) -> null (29 + 33 rows affected).
4. contract_type: case/whitespace variants canonicalized to 3 levels.
5. Exact full-row duplicates dropped: 150 rows removed -> 6000 rows remain.
6. Target `churned` balance after cleaning: 0.0867 (520 positive / 6000).
7. Removed-row check: duplicate rows' churn rate {'0': 5618, '1': 532} - duplicates are spread across both classes (24 churned / 126 not), not concentrated.
