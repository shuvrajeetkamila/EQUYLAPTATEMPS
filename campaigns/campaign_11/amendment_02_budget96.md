# Campaign 11 — Amendment 02 (budget-96 run authorization)

**Date:** 2026-10-14
**Filed:** after the frozen screen runs (`c11_shuf_42`, `c11_shuf_123`,
Actions run `38713676113`) and before the 96-step budget run exists.
**Scope:** executes exactly the run the frozen experiment matrix
marked `PLANNED_AMENDMENT_REQUIRED`. No design element changes.

## Authorization

Run `c11_shuf_42_96`: condition B (within-tensor permuted donor
sector, shuffle seed 20241014), training seed 42, 96 steps, otherwise
identical to the frozen screen configuration
(`experiment_registry_budget96.json`).

## Preconditions observed (from imported, SHA-verified screen records)

- Control validity (frozen criteria in `control_design.md`): **met**.
  All 42 eligible tensors report `multiset_preserved`; the trained
  graft relative update is 0.561 (seed 42) / 0.592 (seed 123), above
  the 0.05 activity floor; bypass and zero ablations are finite and
  ordered above the intact child; `parent_weights_unchanged` is true.
- The primary 32-step comparison is fully recorded; per the
  preregistration this 96-step run is the pre-planned budget
  comparison (intact stored `c10_out_pre_42_96` macro 7.8392 vs this
  run), reported as secondary evidence. It cannot rescue a failed
  primary gate; it tests whether any learned-structure advantage
  grows or vanishes with budget, as in Campaigns 9–10.

## What does not change

Gates, outcome mapping, shuffle seed, pairing, sector, data, and
analysis are exactly as frozen in `preregistration.md`. Execution
uses the commit-tag trigger documented in Amendment 01.
