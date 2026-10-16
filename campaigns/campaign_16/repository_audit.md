# Campaign 16 — Repository Audit (Stage 1)

**Date:** 2026-10-15. **Starting state verified:** local HEAD =
remote HEAD = `67c415fcf7a107a70b8a8848f6f1d988fcaa8e86`
(Campaign 15 final), working tree clean, PR #1 OPEN unmerged.
Campaign 15's stated commits (preregistration `e5bb0f5`, results
`84f94ab`, final `67c415f`) and CI (`38868830250` success) match
the live repository.

## Verified in source code

- **Connectors:** old = `DenseSeam` (LayerNorm→Linear, Xavier;
  Campaign 13/15 condition adds `natgeom_in_trained_out`
  initialization); new = `ResidualDenseSeam`
  (`x + Linear(LayerNorm(x))`, zero-initialized projection —
  bitwise identity verified at runtime in Campaigns 14–15).
  Both expose exactly **667,008** trainable parameters (two
  576-wide seams); parent weights frozen; trainable-contract
  asserted in code. **Optimizer settings are shared**: AdamW
  lr from registry, weight decay 0, clipping 1.0, batch 1 — no
  per-connector optimizer difference exists in code.
- **Training** (`fit_resumable`): data order from a dedicated
  generator seeded per experiment (exactly paired across
  conditions — verified in Campaign 15 traces); scheduled
  evaluation via segment-driving (interrupt/evaluate/resume with
  hash-verified state). Every training step already logs the
  global gradient norm **before clipping** and per-parameter
  gradient norms (`gradient_summary`) — the Campaign 16 gradient
  diagnostics need no invasive instrumentation.
- **Evaluation:** `evaluate_encoded` token NLL; macro = mean of
  language/math group means; **lower is better** (cross-entropy
  loss). Data: GSM8K-train dev 16 + WikiText-2 dev 16, data
  seed 42. Held-out untouched.
- **Baseline learning rate** (Campaign 15 frozen registry):
  **3e-4** → sweep levels 9e-5 (0.3×), 3e-4 (1×), 9e-4 (3×).

## Verified in saved raw records (recomputed independently)

- Campaign 15 reproduces exactly from its SHA-verified records:
  delta0 mean +1.9234; delta32 = [−1.0046, −0.8541, −1.1111]
  (mean −0.9899); delta96 mean +0.1575. Outcome TRAINING_DEPENDENT
  stands as published.
- Per-step diagnostics already present in Campaign 15 traces:
  gradient clipping (threshold 1.0) is active on **100% of
  steps** (pre-clip norms ≈10–14); new-connector median gradient
  norms in steps 1–32 are ≈15–25% **smaller** than the old
  connector's at every seed — a candidate (not yet causal) clue
  for the slower early trajectory. Campaign 16 tests the
  learning-rate prediction this implies under preregistration.

## Cannot be verified / open questions

- Whether the gradient-scale gap explains the reversal (the
  sweep + frozen Gate E signature decide).
- Per-checkpoint (not just final) parameter-update magnitudes
  were not recorded in Campaign 15; Campaign 16's runner adds
  detached snapshots between segments (no change to training
  mathematics; covered by the segmentation equivalence test from
  Campaign 15 plus a Campaign 16 synthetic check).
- No scheduler/warmup exists to preserve (constant LR in the
  library) — the brief's schedule concern does not apply; the
  multiplier design is well-defined.
