# Cleaning log (Stage 1, stateless only)

Raw rows: 6150. Cleaned rows: 6000.

- drop exact duplicate rows: 150 removed, 6000 remaining
- contract_type: mapped 11 spelling variants to 3 canonical levels
- total_charges: -999 sentinel -> NaN (360 rows)
- age: impossible values (>120) -> NaN (32 rows)
- signup_date: parsed as date (explicit format)

Target after cleaning:
churned
0    5480
1     520

churned
0    0.913333
1    0.086667

Removed-row concentration check (dup rows vs class): duplicates were exact copies (same customer_id + all fields), dropped rows are identical to a kept row so class balance is unaffected by construction.
