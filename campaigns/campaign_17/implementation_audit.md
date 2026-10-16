# Campaign 17 — Implementation audit

## What was built

- `src/equylaptatemps/native_placement.py`:
  `build_native_depth_child(...)` assembles a `SurgicalChild` whose
  executed blocks are an explicit **permutation** of the host's
  blocks, split into prefix/graft/suffix at the declared graft span
  (execution coordinates). `validate_execution_order` rejects any
  order that is not an exact permutation of `range(depth)` — deletion
  and duplication are unrepresentable through this interface. Graft
  blocks may come from a separately loaded candidate instance;
  per-block shape signatures are validated against the host sector
  and incompatible candidates raise before assembly. The function
  returns a structural manifest (per-block identity, original/final
  indices, source role, tensor fingerprint, parameter count, action)
  and never reshapes, pads, renames, or reinitializes a tensor.
- `scripts/campaign17_diagnose.py`: single diagnostics job for modes
  A/B/C/D1/D2, reusing the frozen Campaign 14 dev panel, collate /
  predict / evaluate path, and the PR-comment bundle mechanism.
- `.github/workflows/campaign17-diagnose.yml`: push-tagged
  (`[campaign17-diagnose]`), CPU-only, same pins as Campaign 14.
- `tests/test_campaign17_native_placement.py`: permutation legality
  (duplication/deletion rejected), identity-replacement output
  equivalence on tiny synthetic parents, exchange manifest/depth,
  incompatible-candidate rejection, parameter-total identity, and
  continued availability of the historical constructor.

## Deviations and notes

1. `SurgicalChild.structure_report()` derives
   `removed_recipient_blocks` from the *recipe*, not from the
   executed blocks; for engine-built children this field is not
   meaningful. Campaign 17 therefore treats the placement manifest
   plus the tensor audit as authoritative and does not use that field
   for engine modes. Historical code was not modified.
2. The engine requires a non-empty graft span (the `SurgicalChild`
   constructor dereferences `graft.layers[0]`); all frozen modes
   comply.
3. Mode B (historical path) has no engine manifest by construction;
   its placement manifest in `placement_manifest.json` is synthesized
   by the publisher from the recorded execution order, graft span,
   structure report, and the 270/270 tensor-identity audit, and is
   labeled as such. No fingerprints were fabricated for it.
4. The historical constructor deletes recipient blocks 11–12 and
   duplicates sector blocks 24–29 for the registered campaign recipe;
   it remains available unchanged for Mode B and all prior campaigns.

## Verification before the run

- 297 offline tests passed (291 prior + 6 new), `ruff check` and
  `ruff format --check` clean on all touched files, CI
  `38899026184` success on the implementation commit.
