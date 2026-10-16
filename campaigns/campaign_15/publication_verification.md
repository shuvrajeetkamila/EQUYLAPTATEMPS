# Campaign 15 — Publication Verification

- Repository: `shuvrajeetkamila/EQUYLAPTATEMPS`
- Published branch: `arena/97efdf4a-equylaptatemps`
- Preregistration commit: `e5bb0f5` (frozen before any Campaign 15 run)
- Implementation commit: `fe3a568` (segment-driven runner, registries, workflows, tests)
- Final results commit: `84f94ab6505d74ac62da5ef97e0b3ee7b1fb62e6`
- Local HEAD at verification: `84f94ab6505d74ac62da5ef97e0b3ee7b1fb62e6`
- Remote HEAD (git ls-remote): `84f94ab6505d74ac62da5ef97e0b3ee7b1fb62e6` — **match**
- Changed files: campaigns/campaign_15/ (audit, preregistration, evidence,
  reports, imported SHA-verified records), scripts/campaign15_runner.py,
  scripts/publish_campaign15.py, tests/test_campaign15_*.py,
  .github/workflows/campaign15-*.yml, root summary. No historical
  campaign files modified; no weights or checkpoints committed.
- CI: workflow run `38868671803` "Offline surgery checks" — **success**.
- Tests: 288 passed locally (0 failed/skipped); ruff check + format clean.
- PR #1: OPEN, unmerged, retitled "Campaigns 1-15 complete — connector
  ranking is training-dependent: healthy start wins init + step 96,
  historical wins step 32".
- Campaign jobs (all success, SHA-verified imports): regression
  38864353328, smoke 38864353272, panels 38865626143 / 38865628701 /
  38865631168.
- Uncommitted/unpublished files at verification: none besides this file,
  which is committed immediately after writing.
- Publication blockers: none.
