# Campaign 18 — Publication verification

- Results commit: `fd0cd88a2f7d797854175cb4c6c223a5da22c8fe`
  (this file's own commit follows and is verified identically; the
  final published SHA is the HEAD containing this file).
- Local HEAD equals remote `arena/97efdf4a-equylaptatemps` at the
  results commit (verified via `git ls-remote`); CI on it:
  `38914691473` success ("Offline surgery checks"); the Campaign 18
  workflow correctly skipped (no experiment tags on result commits).
- Experiment runs: map `38907871417` (success, dev+validation,
  53/53 scored) and held-out `38911632615` (success, 30/30 scored,
  single recorded opening after predictor freeze); both bundles
  SHA-verified on import. CI on the held-out freeze commit:
  `38911641942` success.
- PR #1: open, unmerged, retitled for Campaigns 1–18. No merge,
  force-push, branch deletion, or history rewrite. No weights or
  secrets committed.
- Label: `PLACEMENT_AWARE_GRAFT_DESIGN_READY`; Campaign 19 NOT
  authorized (Gate G6 not granted). Capability held-out:
  `NOT_RUN_NO_PROMOTION`. Failures: none (failure_registry.json).
- Checksums: `checksums.sha256` (54 files, verified before commit).
