# Campaign 12 — Implementation Audit

**Date:** 2026-10-14
**Preregistration freeze:** commit `3f5df5f` (before any C12 run)
**Jobs:** validation `38726596264`; 32-step matrix `38726922262`;
96-step panel `38738479882`. All bundles SHA-256-verified on import
(`639431d5…`, `346a3f41…`, `a46764db…`).

## Implementation

- `row_shuffled_sector_control` added to
  `src/equylaptatemps/controls.py`, mirroring Campaign 11's
  full-tensor control: deep-copies the graft, permutes each eligible
  2D matrix independently within each output row (single generator,
  seed 20241014), verifies per-row sorted multisets exactly, records
  SHA-256 before/after per tensor, excludes the 12 RMSNorm gains,
  never touches the donor parent.
- `scripts/campaign12_runner.py` derived from the Campaign 11
  runner (new condition branch; campaign-12 registry/results
  naming). `scripts/campaign12_validate.py` constructs all four
  children with zero training for pre-training validation.
- `scripts/publish_campaign12.py` computes every published number,
  validity check, gate, and the classification mechanically.
- Execution via the established commit-tag trigger (Campaign 11
  Amendment 01 applies; dispatch endpoint still unavailable).

## Verification highlights

- **Control reproduction:** Campaign 12's condition-C sector is
  bit-identical to Campaign 11's: all 54 recorded tensors share
  identical SHA-256 before and after permutation across campaigns.
- **Fresh-run replication:** new intact runs reproduce stored
  Campaign 10 values closely (32 steps seed 42: 8.3670 fresh vs
  8.3665 stored; 96 steps seed 42: 7.8392 vs 7.8392).
- Pre-training validation (before any training): all four children
  finite, identical graft parameter counts (21,240,576), parents
  unchanged; B/C multiset checks pass at construction.
- All 20 confirmatory runs COMPLETED; every record shows
  `parent_weights_unchanged: true` and `test_metrics: null`.
- 8 new tests (row-shuffle properties, gate mechanics, results
  tracing) plus the full suite pass; lint clean.

## Deviations and amendments

- **Amendment 01 (validity criterion):** the publisher initially
  implemented a stricter ablation-ordering check than the frozen
  preregistration's criterion (4) ("finite and above intact"). Two
  within-row runs have bypass worse than zero while both ablations
  are finite and far above intact. The publisher was corrected to
  the frozen text; the stricter ordering is reported descriptively
  (holds in 10 of 12 scrambled runs). Classification under the
  miscoded check would have been D; under the frozen criterion the
  controls are valid and the outcome is C. The gate fails under
  either reading. Full disclosure in
  `amendment_01_validity_criterion.md`.
- No other deviations. No run rerun, excluded, or replaced.
