# Stage 1 cleaning log

rows read: 4320
exact duplicate rows: 0 (none dropped, count=0)
listing_id duplicates: 0
property_id: 4000 unique / 4320 rows -> 320 repeat-sale rows (GROUP_KEY for Stage 2 split)

sentinel/impossible -> null (stateless):
  sqft: 50 and 100000 treated as sentinel -> 21 rows nulled (now 21 missing, 0.49%)
  lot_size_sqft: 9999 treated as sentinel -> 1 rows nulled (now 866 missing, 20.05%)
  year_built: 1800 and 2090 treated as sentinel (isolated spikes, gap to nearest real value >50yr) -> 22 rows nulled (now 22 missing, 0.51%)
  garage: spelling variants {Yes,yes,Y,YES}->Yes, {No,no,N}->No; blank stays missing (now 60 missing, 1.39%)

rows remaining after Stage 1: 4320 (no rows dropped; impossible values nulled in place, not removed)
