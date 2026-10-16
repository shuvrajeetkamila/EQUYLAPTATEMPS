# Campaign 19 execution — Repository preflight (Stage 0)

## Verified directly

- Branch `arena/97efdf4a-equylaptatemps`; local HEAD restored to
  remote `1ae9be94127f2435b03263fbdfd48bb7539d8812` (Campaign 19
  preparation publication); tree clean; PR #1 open, unmerged.
- Frozen preregistration retrieved from commit `b0599b1`
  (`git show b0599b1:campaigns/campaign_19/config/preregistration.json`)
  and re-read in the working tree: recipient
  `HuggingFaceTB/SmolLM-135M@1d461723…`, donor
  `state-spaces/mamba-130m-hf@1e76775f…` (safetensors SHA-256
  `1a5ed29c…` in the frozen registry), graft [11,13)→[11,13),
  LR 3e-4, 96 steps, seeds {42,123,31415}, advantage ≥ 0.25 at
  step 32, ablation ≥ 50%, preservation ≤ +4.50, D0 margin ≥ 0.05
  (exploratory), held-out = new disjoint slice only.
- Run count reconciliation (Stage 2): the frozen matrix defines
  **10 training runs** — C/D/E × 3 seeds (9) + H × seed 42 (1).
  A, B, D0, F, G are evaluation-only (F/G derive from trained C
  checkpoints). This matches the preparation summary's "ten
  training runs"; no discrepancy.
- Preparation CI `38937301822` success (verified during
  preparation); 308 offline tests at freeze.
- Campaign 18 readiness: `PLACEMENT_AWARE_GRAFT_DESIGN_READY`, G6
  authorization now supplied by the Campaign 19 execution brief.

## Execution-environment note

Real weights run in GitHub Actions (no local GPU/HF access), per
Campaigns 14–18 convention. Checkpoint hashes are verified in-job
against the frozen registry via snapshot provenance before any
scoring; a mismatch aborts the job (recorded as preflight failure).
