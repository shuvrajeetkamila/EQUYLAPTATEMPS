# Campaign 7 Integrity Report

- **Pinned revisions enforced:** host `1e76775f…`, donor `1d461723…`; acquisition
  refuses non-safetensors weights and any revision mismatch
  (`scripts/prepare_parents.py::acquire`).
- **Parents frozen:** the runner fingerprints both parents before and after all
  runs and raises on any change; both jobs report `parents_unchanged: true`.
  Only the two declared dense seams (888,768 params) were trainable;
  `assert_trainable_contract` held for pretrained and twin children.
- **Twin fairness:** identical architecture, shape, seams, data order, budget,
  and seed; the only difference is donor-sector initialization
  (pretrained vs native-init seed 1234). Twin grafts verified frozen, same
  parameter shapes (`tests/test_campaign07_llama.py`).
- **Evidence chain:** both jobs posted gzip+base64 evidence bundles to PR #1;
  imported bundles verified against their embedded SHA-256
  (screen `d568e599…`, replication `3c59160e…`) before any number was used.
- **No TEST data** was used for training, selection, or thresholds; held-out
  remains `NOT_RUN_NO_PROMOTION`. Failed/preserved artifacts: none of the four
  registered runs failed; the probe-v1 log-retrieval failure and the
  replication-workflow commit race are documented in `failure_analysis.md`.
