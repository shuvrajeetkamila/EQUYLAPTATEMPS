# Campaign 12 — Repository Audit (Phase 0)

**Date:** 2026-10-14
**Repository:** `shuvrajeetkamila/EQUYLAPTATEMPS`
**Branch:** `arena/97efdf4a-equylaptatemps` (authoritative project
branch for this session; PR #1 is open against it and unmerged)

## Starting state (verified, not assumed)

- Remote URL: `https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git`
- Remote HEAD at audit start: `a1566a28ecf5d20c05a92b17250b2de0cef2aabc`
  ("Campaign 11 complete: Outcome C").
- The fresh local checkout presented the known session-restore
  artifact (HEAD at the initial commit `ecc8b0d`, partial tree).
  Recovery was non-destructive: `git fetch origin`,
  `git reset --mixed origin/arena/97efdf4a-equylaptatemps`,
  `git restore --source=HEAD --worktree --staged .`. Afterwards
  `git status --porcelain` = 0 and local HEAD = remote HEAD.
  No history rewritten, nothing force-pushed, no unrelated work lost.
- PR #1: OPEN, title "Campaigns 1-11 complete — learned-structure
  test: valid scrambled control, no reliable donor advantage
  (Outcome C)". Not merged (must remain so).
- Latest CI at audit start: `38717263219` "Offline surgery checks"
  completed/success on `a1566a2`.

## Historical evidence checked

- Campaign 11's three headline comparisons were recomputed from the
  raw imported records (`results.json` → `validation.groups` token
  NLLs): A 8.3665 vs B 8.6808 (32 steps, seed 42); A 8.3132 vs B
  8.3098 (32/123); A 7.8392 vs B 7.7141 (96/42). All match the
  published table exactly. Every record shows
  `parent_weights_unchanged: true` and `test_metrics: null`
  (held-out never opened). Published outcome C and
  `control_valid: true` trace to `campaigns/campaign_11/results.json`.
- Campaign 11 control: `shuffled_sector_control` in
  `src/equylaptatemps/controls.py` (within-tensor permutation, seed
  20241014, 42 eligible 2D matrices, 12 RMSNorm gains excluded) —
  reused unchanged as Campaign 12 condition C.
- Runner lineage: `scripts/campaign11_runner.py` (conditions selected
  per-experiment via `shuffled_sector` / `random_sector` flags;
  `natgeom_in_trained_out` initialization; bypass/zero ablations and
  graft-activity diagnostics recorded for every run; parents
  fingerprinted and re-verified unchanged).
- Canonical gates (from Campaigns 7–11 preregistrations, unchanged):
  per-seed donor benefit thresholds (≥2% at seed 42, ≥1% at seed
  123), usefulness gate G-C (child language NLL ≤ 1.50 × host
  4.559452953338623; macro ≤ 6.00), promotion only if all gates +
  control validity + causal ablation + integrity pass, else held-out
  `NOT_RUN_NO_PROMOTION`. Outcome letters A/B/C/D as defined in
  Campaign 11's preregistration.
- Model revisions (pinned, unchanged): recipient
  `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`;
  donor `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`;
  sector [24,30), cut [11,13).

## Known interpretation hazards carried forward (binding)

- A near-silenced control (graft update ≈0.001, as the raw random
  twin showed in Campaigns 9–10) is not evidence for the intact
  donor; activity must be reported per condition and validity gates
  applied before any comparison counts.
- A seed-level sign reversal is not robustness; all seeds reported.
- Ablations/activity are diagnostics, never success criteria.

## Compute limitations

Local sandbox: CPU-only, no Hugging Face access; real-weight runs
execute on GitHub Actions (ubuntu-22.04, 2 threads, float32) via the
repository's commit-tag workflow trigger (Campaign 11 Amendment 01
documents why: the sandbox token's dispatch endpoint returns 403).
Campaign 11 timings: ~4–5 min per 32-step run, ~11 min per 96-step
run, plus ~5 min environment/model setup per job.

## Verification commands used

`git ls-remote origin arena/97efdf4a-equylaptatemps`;
`gh pr view 1`; `gh run list --branch arena/97efdf4a-equylaptatemps`;
local recomputation of Campaign 11 macros from raw records
(plain Python over the imported JSON evidence).
