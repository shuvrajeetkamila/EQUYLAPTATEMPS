# Campaign 7 Screening Analysis (from raw records, jobs `38542143640` + `38544932532`)

Host language dev NLL (reference): **4.5595**. Macro = mean(language, math) dev NLL.

| Run | Macro | Language (×host) | Bypass | Zero | Graft rel. update |
|---|---|---|---|---|---|
| c07_dense_pre_42 | 8.5719 | 8.657 (1.899×) | 10.930 | 44.072 | 54.28 |
| c07_dense_rand_42 (twin) | 7.5349 | 7.926 (1.738×) | 7.651 | 14.366 | 0.472 |
| c07_dense_pre_123 | 9.4023 | 10.108 (2.217×) | 10.509 | 10.211 | 57.59 |
| c07_dense_rand_123 (twin) | 7.8900 | 8.116 (1.780×) | 7.987 | 12.239 | 0.472 |

**Frozen gate evaluation**

- G-B (pretrained ≤ 0.98 × twin): seed 42 → 8.5719 vs 7.3842 **FAIL** (benefit
  −13.76%); seed 123 → 9.4023 vs 7.7322 **FAIL** (benefit −19.17%). The negative
  direction replicates and grows; this is not seed noise.
- G-C (language ≤ 1.50× and macro ≤ 6.00): **FAIL** at both seeds.
- Donor-use diagnostics: bypass/zero ablations and a graft relative update of
  ≈54–58 (vs ≈0.47 for twins) show the pretrained sector dominates the child's
  signal path — the child genuinely *depends* on it. Under the frozen
  interpretation (Campaigns 3–6), dependence is not benefit: the matched random
  sector is simply better under identical training.

**Decision:** no promotion; held-out `NOT_RUN_NO_PROMOTION`; classification
**Outcome C** (see final report). Numbers above are regenerated from the
imported records by `scripts/publish_campaign07.py` and traced by
`tests/test_campaign07_results.py`.
