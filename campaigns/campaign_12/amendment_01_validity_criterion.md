# Campaign 12 — Amendment 01 (validity-criterion implementation discrepancy)

**Date:** 2026-10-14
**Filed:** after all 20 confirmatory runs completed and were imported,
during publication. **No threshold, gate, or outcome definition is
changed by this amendment.** It corrects publisher code to match the
frozen preregistration and discloses the discrepancy.

## The discrepancy

The frozen preregistration (`preregistration.md`, commit `3f5df5f`,
before any Campaign 12 run) defines control-validity criterion (4) as:

> "(4) bypass and zero ablation macros finite and above intact"

`scripts/publish_campaign12.py` as first written implemented a
stricter check copied from Campaign 11's publisher
(`intact < bypass < zero`). Under that stricter check, two
within-row runs (`c12_row_31415`, `c12_row_42_96`) were flagged
invalid because bypass was *worse* than zero (e.g. intact 8.4337,
zero 21.1751, bypass 33.1513) — while both ablations were finite and
far above intact, i.e. criterion (4) as frozen is satisfied, and the
graft is unambiguously causally load-bearing in those runs.

## Resolution (protocol-faithful)

The frozen preregistration is the authoritative artifact; code must
implement it, not exceed it. The publisher is corrected to implement
criterion (4) exactly as written (finite, and both ablations above
intact). The stricter bypass<zero ordering is retained as a
*reported descriptive statistic* per run, not a validity gate; it
holds in 10 of 12 scrambled-control runs and its two exceptions are
reported in the final report.

## Materiality and disclosure

- Under the miscoded stricter check, the mechanical classification
  would have been D (inconclusive). Under the frozen criterion, the
  controls are valid and the classification is C (the
  learned-structure gate fails regardless of this amendment).
- The learned-structure gate, usefulness gate, and all benefit
  numbers are unaffected by this amendment. No run is excluded
  under either reading.
- This amendment was written after results were seen; that timing
  is disclosed here rather than hidden. The justification rests
  solely on the pre-run frozen text, quoted above.
