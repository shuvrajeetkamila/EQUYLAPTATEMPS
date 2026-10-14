# Campaign 8 Test Report

- New: `tests/test_campaign08_calibration.py` (3 tests) — calibration is
  deterministic, interface-only, statistics-required, and (Amendment 01) the
  noscale variant skips/records out-of-range gains instead of raising.
- New: `tests/test_campaign08_results.py` — every published macro recomputed
  from raw records across Campaigns 7–8; all six benefits negative; Outcome C
  traced.
- Inherited coverage: Llama loading/mapping/intactness, twin fairness,
  budget/registry parity, held-out protection, summary regeneration
  (Campaigns 1–7 suites unchanged).
- Full suite + whole-repo `ruff check`/`ruff format --check` at publication:
  see final report and CI on PR #1. Tests prove implementation correctness
  only — not capability; the capability claim is the (negative) experiment.
