# Data integrity report

- Donor frozen in all 10 runs (`donor_fingerprint_unchanged: true`);
  parents unchanged at job end in both jobs.
- All 10 registered runs present with per-example step-32/96 scores;
  no seed skipped or replaced (attempt 1 failure F-C19-001 produced
  no scores and is preserved separately).
- Scores recomputed by the publisher from SHA-verified bundles;
  consistency checks in `report_consistency_check.json`.
- Held-out: never acquired; access log empty.
