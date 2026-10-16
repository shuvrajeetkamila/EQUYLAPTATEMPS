# Campaign 19 — Publication verification (execution)

- Results commits: `feed89aa5b325419383dc213955477ea8cf0374b`
  and README/checksum correction `334876abfb5069b8537b6481c77aab03ad92a240`
  (this file's commit follows; final SHA is the HEAD containing it).
- Local = remote verified at `334876a…`; CI `38965254935` success.
- Experiment runs: gates `38947055805` (D0 PASS, B bitwise PASS),
  training attempt 1 `38947619114` (failed pre-results, F-C19-001),
  training `38950005531` (10/10 runs, SHA-verified bundles).
- Verdict: `DONOR_ADVANTAGE_NOT_SHOWN` /
  `VALID_NEGATIVE_NO_DONOR_ADVANTAGE`. Held-out:
  `NOT_RUN_NO_PROMOTION` (never acquired). PR #1 open, unmerged,
  retitled. No merge/force-push/history rewrite; no weights
  committed. Checksums: `checksums.sha256` (109 files).
