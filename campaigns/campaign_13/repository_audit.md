# Campaign 13 — Repository Audit (Phase 0)

**Date:** 2026-10-14
**Repository:** `shuvrajeetkamila/EQUYLAPTATEMPS`
**Branch:** `arena/97efdf4a-equylaptatemps`
**Campaign 13 starts from:** `88f373de6f70a5ca32c2d092539d3c57a535d749`
("Campaign 12 complete: Outcome C"), verified as remote HEAD.

## Starting state (verified, not assumed)

- Remote URL: `https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git`.
- The fresh checkout again presented the known session-restore
  artifact (local HEAD at the initial commit, partial tree);
  recovered non-destructively (`git fetch`, `git reset --mixed
  origin/arena/...`, `git restore --source=HEAD --worktree
  --staged .`). Post-recovery: `git status --porcelain` = 0,
  local HEAD = remote HEAD = `88f373d`. No history rewritten.
- PR #1: OPEN, unmerged ("Campaigns 1-12 complete — structure-scale
  test … (Outcome C)"). Latest CI: `38758810215` "Offline surgery
  checks" completed/success.
- Campaign 12 evidence present locally and remotely: 12 screen runs,
  8 budget runs, validation bundle, all published artifacts.

## Independent review of Campaign 12 Amendment 01 (required by brief §2.2)

Verified directly against source files, git history, and raw records:

1. **Frozen rule.** `campaigns/campaign_12/preregistration.md` at the
   freeze commit `3f5df5f` (committed before any Campaign 12 run)
   defines validity criterion (4) verbatim as: *"bypass and zero
   ablation macros finite and above intact."*
2. **Original verdict code.** At implementation commit `b6cced4`,
   `scripts/publish_campaign12.py` line 122 implemented the stricter
   `intact < bypass < zero` under the key
   `ablations_finite_and_ordered` — stricter than frozen. The
   discrepancy was real and was in the code, not the protocol.
3. **Discovery.** During final publication, after all 20 runs were
   imported; the amendment is dated and discloses this timing.
4. **Evidence.** Independently recomputed from raw records: exactly
   two runs differ between readings — `c12_row_31415` (intact
   8.4337, bypass 33.1513, zero 21.1751) and `c12_row_42_96`
   (7.7717, 26.5689, 24.9973). In both, ablations are finite and
   far above intact, so the frozen criterion is satisfied; only
   the extra bypass<zero ordering fails (bypass can be worse than
   zeroing when the trained seams amplify a bypassed signal —
   mechanistically unsurprising).
5. **Preservation.** The strict diagnostic was not deleted: it is
   recorded per run as `bypass_below_zero_ordering` in
   `control_validation.json` (2 of 10 scrambled runs), and a test
   asserts the count of exceptions is exactly 2.
6. **Consistency of correction.** `git diff b6cced4..88f373d` for the
   publisher is confined to job IDs, bundle SHAs, and the validity
   block. No benefit value, gate input, or run inclusion changed.
7. **Outcome C follows mechanically.** Recomputed from raw records:
   the learned-structure gate is False (seed 123 vs C = −0.57% <
   1%; seed 31415 negative vs both controls) *regardless* of the
   validity reading. Under the stricter misreading the outcome
   would have been D; the amendment changes only that label, and
   the final report discloses both readings.

**Verdict:** the amendment is correct and honestly documented; the
Campaign 12 baseline is reliable and Campaign 13 may proceed on it.
One process lesson is adopted for Campaign 13 (brief Part 9): the
publisher will implement validity *exactly* as the frozen text, a
unit test will pin the rule against hand-computed examples before
training, and both the frozen criterion and any stricter diagnostic
will be reported per run.

## Pipeline constraints discovered (load-bearing for pair selection)

- `Recipe.validate` deliberately **rejects same-family/shared-lineage
  parents** ("Cousin parents rejected") and `load_parents` rejects
  identical tokenizer fingerprints — the cross-family programme's
  "stranger check". A same-family positive control therefore
  requires one explicit, preregistered, defaulted-off opt-in
  (`Recipe.positive_control`), never a silent bypass. Design in
  `model_pair_selection.md`.
- Parent acquisition is allowlist-based (`REGISTRY` in
  `src/equylaptatemps/recipe.py`) with exact 40-hex revisions
  enforced; new parents must be added to the reviewed registry.
- Sandbox has no Hugging Face access; exact revisions for any new
  checkpoint must be resolved by an exploratory Actions probe and
  frozen into the preregistration before confirmatory runs.
