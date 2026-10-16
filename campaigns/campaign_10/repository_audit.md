# Campaign 10 Repository & Evidence Audit (Phase 0, 2026-10-14)

## Live state

- Branch `arena/97efdf4a-equylaptatemps` (session-fixed), HEAD restored
  non-destructively to the verified remote head
  `dc313d3fae70cfa78c6876cfe3f241846a7284df` after the recurring
  session-restore artifact (local base-commit tree, `.venv` absent; nothing
  lost — recovery from remote, tree verified clean).
- PR #1 OPEN: "Campaigns 1-9 complete — donor-native geometry, replicated
  donor-specific effect, Outcome B"; CI on `dc313d3`: success (3.10/3.11).
- Resources unchanged: sandbox has no GPU and ~3.8 GiB RAM; real-parent runs
  execute on GitHub Actions CPU runners via opt-in workflows (proven in
  Campaigns 2–9). torch 2.5.1 / transformers 4.48.3.

## Campaign 9 verification (from primary records, not the summary)

All ten published macros recomputed from raw `results.json` files match
`campaigns/campaign_09/results.json` exactly; the brief's table
(−13.76/−19.17/+9.49/+10.37/+11.49%) is confirmed. Outcome B; held-out
`NOT_RUN_NO_PROMOTION`. No discrepancies found.

## Output-pathway audit (Step 2.2 — the question Campaign 10 turns on)

From `campaigns/campaign_09/screen_38578925047/runs/*/initialization.json`
and `src/equylaptatemps/alignment.py`: in Campaign 9 **both** seams were
initialized by `donor_native_geometry` — the output seam was whitened by the
child-output covariance and colored by the recipient-native output
covariance (plus an in-range RMS gain ≈1.55), then trained. The output
pathway is fully trainable in all conditions (888,768 trainable seam
params; identical in pretrained and twin), receives gradients normally, and
differs between B/C/D of Campaign 10 only by the change frozen below: the
output seam will start from the generic trained initialization (Campaign-7
`DenseSeam` default) with **no geometry preconditioning**, while the input
seam keeps Campaign 9's geometry exactly. Donor [24,30) executes natively
(`backbones.py`), frozen, fingerprinted per run.
