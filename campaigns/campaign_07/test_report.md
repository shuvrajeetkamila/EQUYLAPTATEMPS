# Campaign 7 Test Report

- New: `tests/test_campaign07_llama.py` (7 tests) — SmolLM config loading,
  provenance-registry entry, Llama block/norm construction (incl. GQA shapes),
  synthetic parent extraction, Mamba→Llama forward/backward with seam-only
  gradients, matched frozen random-twin control, `empty_child` roundtrip.
- New: `tests/test_campaign07_results.py` — published macros recomputed from
  the imported raw `results.json` records; gate outcomes and Outcome C traced.
- Full suite at publication: see final report / CI on PR #1 (`ruff check .`,
  `ruff format --check .`, pytest on Python 3.10/3.11). All Campaign 1–6 tests
  unchanged and passing; whole-repo lint clean.
