# Campaign 17 — Implementation Design

**New module `src/equylaptatemps/native_placement.py`:**
`build_native_depth_child(recipient_parent, *, execution_order,
graft_span, seam_mode, recipe, donor_parent=None)` —

- Validates `execution_order` is a permutation of `range(depth)`
  (deletion/duplication are *unrepresentable*, not merely checked
  after the fact) and `graft_span` names the declared replacement
  region in execution coordinates.
- Assembles prefix/graft/suffix `StackSegment`s from the actual
  parent block modules (prefix/suffix from the recipient
  instance; graft from the donor-role instance when supplied —
  the honest simulation of a component swap — else recipient).
- Verifies per-block parameter-shape signatures against the
  segment's first block and the configured hidden width; raises
  on any mismatch (no silent truncation/padding/renaming).
- Installs `residual` seams via the established
  `install_interfaces` path or bypasses them (`nn.Identity`).
- Returns the child plus a structural manifest (per-block stable
  identity, fingerprints via `module_fingerprint`, actions,
  duplicated/removed lists — empty by construction and verified).

**Modes:** A native (reference); B historical-path exact
reconstruction ([24,30)←[24,30), bypassed) for Campaign 14
continuity; C identity replacement through the new engine
(identity order, graft [11,13) from the donor-role copy, residual
seams); D1 long-range sector exchange (swap [11,13)↔[24,26),
bypassed); D2 adjacent exchange (swap blocks 15↔16, bypassed).
No training anywhere in Campaign 17: the preregistered questions
concern initial structural behavior, and Campaign 16 settled the
connector-training methodology; training an exchange would
measure recovery, not placement (Stage 9 rationale).

**Versus the historical path:** `child_from_parents` registers a
(donor sector → recipient cut) replacement whose spans may differ
in length, silently changing depth/order. The new engine's input
is an explicit execution order over the host's own depth; length
change is impossible, and every deviation from identity must be
declared in the manifest.
