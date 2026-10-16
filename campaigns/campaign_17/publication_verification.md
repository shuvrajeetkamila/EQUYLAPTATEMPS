# Campaign 17 — Publication verification

- Results commit: `7d1babb072441da38254c30e69603f302af94d42`;
  cleanup commit (this file's parent): `14171ff0e7dc4b23b14fe0ce3d285e705e65d642`.
- Local HEAD equals remote `arena/97efdf4a-equylaptatemps` (verified
  via `git ls-remote` before this file was added; this file's own
  commit is verified the same way immediately after push and its CI
  inspected — the final SHA is the HEAD containing this file).
- Diagnostics evidence: GitHub Actions run `38899506528` (success,
  commit `5a8ebf4`), bundle SHA-verified on import into
  `diag_38899506528/`. CI on the results commits: `38900514850`
  success ("Offline surgery checks"); the Campaign 17 diagnostics
  workflow correctly skipped (tag absent) on result commits.
- PR #1: open, unmerged, retitled for Campaigns 1–17. No merge, no
  force-push, no branch deletion, no history rewrite.
- Label: `INTERFACE_REPAIRED_STRUCTURAL_DAMAGE_REMAINS`. Held-out:
  `NOT_RUN_NO_PROMOTION`. No weights, secrets, or private data
  committed. Checksums: `checksums.sha256` (30 files, verified).
- Failure registry: F-C17-001 records the new-workflow trigger lag
  (infrastructure; the frozen diagnostics job ran exactly once,
  under run 38899506528).
