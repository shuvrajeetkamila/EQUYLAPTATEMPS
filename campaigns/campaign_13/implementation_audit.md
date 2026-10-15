# Campaign 13 — Implementation Audit

**Date:** 2026-10-14
**Preregistration freeze:** commit `6e6b41b` (before confirmatory training)
**Jobs:** probe `38766385142` (exploratory), validation
`38767049795`, 32-step matrix `38767422815`, 96-step panel
`38770695036`, self-test `38770694519`. All bundles SHA-256-verified
on import (`3c4c1eaa…`, `9d8cfd22…`, `914bd262…`, `bf953eb2…`,
`13e7c57e…`).

## Implementation

- **Reviewed-registry extension:** `HuggingFaceTB/SmolLM-135M-Instruct`
  added to the parent allowlist (`recipe.py`), documented as
  same-lineage, usable only via the opt-in below.
- **`Recipe.positive_control` opt-in** (default `False`): skips the
  cousin rejection and the distinct-tokenizer requirement, makes
  `stranger_report()` report the truth
  (`different_architectures: false`, mode
  `same_family_positive_control`), and is recorded in every run's
  structure report. Serialized only when enabled, so default
  recipes and previously published artifact audits remain
  byte-identical (the checksum-pinned golden child archive still
  validates). Cross-family behavior is unchanged — the existing
  cousin-rejection test passes unmodified.
- `scripts/campaign13_runner.py` / `campaign13_validate.py` derived
  from the Campaign 12 pair with positive-control recipes;
  `scripts/campaign13_probe.py` (exploratory pair selection);
  `scripts/publish_campaign13.py` implements the frozen validity
  list *verbatim* (pure `ablation_criterion` unit-tested against
  hand-computed examples before training — the Campaign 12 lesson).
- Execution via the established commit-tag trigger.

## Verification highlights

- Probe: donor revision resolved and pinned
  (`fcc320f490e08fdb4b99d935b2c58d40bf35b0d0`); configs
  architecturally identical; sector delta 5.64% mean relative RMS;
  native donor worse than recipient on dev NLL (3.0369 vs 2.7731),
  so the control tests structural detectability, not donor
  superiority (framing frozen in the preregistration).
- Pre-training validation: all four children finite, identical
  graft counts (21,240,576), B/C multisets exact, parents unchanged.
- All 20 confirmatory runs + the F self-test COMPLETED; every
  record: `parent_weights_unchanged: true`, `test_metrics: null`,
  `positive_control: true` in structure.
- 11 new Campaign 13 tests + full suite pass; lint clean.

## Deviations and amendments

None. No run rerun, excluded, or replaced; no threshold touched
after freezing. Two engineering defects were caught and fixed
*before* the freeze (a same-file edit race that briefly dropped
two opt-in edits, and byte-compatibility of default recipe
serialization with the published golden artifact); both are
recorded in the campaign working history and neither touched any
frozen scientific element.
