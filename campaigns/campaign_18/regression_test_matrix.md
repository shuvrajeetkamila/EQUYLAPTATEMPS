# Campaign 18 — Regression test matrix

| Area | Coverage |
|---|---|
| Exact reconstruction / identity | Campaign 17 engine tests + Campaign 18 in-job controls (bitwise) |
| Stable identities, no deletion/duplication/reorder | permutation validator tests; manifest audits in every run |
| Indexing convention & inverse ops | `test_campaign18_map.py` (final-position and inverse-composition tests) |
| Frozen set integrity | sets SHA-256 recomputed in test; 83 disjoint conditions |
| Damage math & bookkeeping | displacement/feature bookkeeping tests; publisher derives all numbers from raw bundles |
| Held-out sealing | `heldout_access_log.json` (one recorded opening after predictor freeze); `heldout_integrity.json` |
| Historical suites | all prior tests unchanged (302 passing at implementation; see `test_report.txt`) |
| Invalid manifests | five probes rejected in-job + offline permutation tests |
