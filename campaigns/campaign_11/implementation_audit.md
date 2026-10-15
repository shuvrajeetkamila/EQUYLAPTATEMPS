# Campaign 11 — Implementation Audit

**Campaign:** 11 — The Learned-Structure Test: Pretrained Donor vs
Statistics-Preserved, Structure-Destroyed Controls
**Date:** 2026-10-14
**Preregistration freeze commit:** `fb90ebb`
**Screen job:** GitHub Actions `38713676113` (bundle SHA-256
`5876079a…2c2`, verified on import)
**Budget-96 job:** GitHub Actions `38715108842` (bundle SHA-256
`b8db0ce9…c13`, verified on import)

## What was implemented

- `shuffled_sector_control` (`src/equylaptatemps/controls.py`):
  deep-copies the six transplanted donor blocks, applies one
  within-tensor permutation (single `torch.Generator`, seed
  **20241014**) independently to each eligible 2D weight matrix,
  leaves 1D tensors (the 12 RMSNorm gains) untouched, and records
  per-tensor name, shape, eligibility, multiset-preservation, and
  SHA-256 before/after in `child.control_record`. The donor parent
  model is never modified (its state dict is deep-copied first).
- `scripts/campaign11_runner.py`: derived from the Campaign 10
  runner; adds the shuffled-sector branch and serializes
  `sector_control` into each run's `initialization.json`.
- Registries `experiment_registry.json` (screen: `c11_shuf_42`,
  `c11_shuf_123`, 32 steps) and `experiment_registry_budget96.json`
  (`c11_shuf_42_96`, executed under Amendment 02).
- Execution via the repository's established commit-tag trigger
  (Amendment 01: the sandbox token's workflow-dispatch endpoint
  returned HTTP 403; no workflow file was changed).

## Verification performed

- 42 eligible tensors per run (7 matrices × 6 blocks:
  q/k/v/o projections, gate/up, down); 12 RMSNorm gain vectors
  excluded and preserved, as preregistered.
- In every condition-B run record: `all_multisets_preserved = true`,
  every eligible tensor's SHA-256 changed (permutation actually
  applied), `shuffle_seed = 20241014`.
- Unit tests (`tests/test_campaign11_shuffle.py`, 4 tests): donor
  parent immutable, exact multiset preservation, deterministic under
  the frozen seed, graft remains frozen, child executes.
- Results-tracing tests (`tests/test_campaign11_results.py`):
  every published macro is recomputed from the raw imported records;
  control-validation fields trace to `initialization.json`.
- Condition A values are the stored Campaign 10 condition-B records
  (jobs `38602501026`, `38606208077`, `38607522188`), labeled
  historical; Campaign 10 was not rerun.
- All runs: `parent_weights_unchanged = true`; held-out test metrics
  `null` in every record (never opened).

## Deviations

- Amendment 01 (trigger mechanism only; filed before any C11 run).
- Amendment 02 (authorizes the frozen `PLANNED_AMENDMENT_REQUIRED`
  96-step run; filed after screen import, before that run existed).
- No scientific deviation: pairing, sector [24,30), cut [11,13),
  interface `natgeom_in_trained_out`, data, budgets, gates, shuffle
  seed, and outcome mapping are exactly as frozen.
