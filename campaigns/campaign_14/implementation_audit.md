# Campaign 14 — Implementation Audit

## What was changed, and why it is safe

**New (opt-in, historical behavior untouched):**
- `ResidualDenseSeam` (`src/equylaptatemps/model.py`):
  `x + Linear(LayerNorm(x))` with the projection weight and bias
  zero-initialized — the exact identity map at initialization.
  Same-width only; a dimension-changing map cannot be the
  identity, so mismatched widths raise. The projection receives
  nonzero gradients at the identity point (zero-init output
  branch, as with ReZero-style gates); the norm's gradients
  vanish only at the exact initial point and appear as soon as
  the projection moves. Verified offline: exact `torch.equal`
  identity, nonzero projection gradient, install path through
  `install_interfaces` ("residual_identity" architecture),
  trainable-contract compliance, and exact equality with a
  bypassed-seam child at initialization
  (`tests/test_campaign14_residual_seam.py`).
- `scripts/campaign14_diagnose.py`: untrained variant matrix
  (V0–V6) with tensor-by-tensor parameter audits, fixed-batch
  logit equivalence, per-example scores, ablations, graft
  activity, runtime/memory.
- `scripts/campaign14_runner.py`: Campaign 13 runner derivation
  adding the `residual_identity` initialization branch (V7/V8).
- Workflows `campaign14-{diagnose,trained}.yml`.

**Code-reading facts that shape the interpretation:**
- `StackSegment.forward` derives RoPE position ids from the
  attention mask (global token positions), identical for every
  segment — relocation does not restart positions; the V1 bitwise
  match confirms the whole native path (masking, RoPE, tied head)
  is reproduced exactly by the surgery machinery.
- `child_from_parents` assembles prefix/graft/suffix from the
  parents' own modules; the audits (273/273 tensors bit-identical
  at native placement; 309/309 at the Campaign 13 placement with
  exactly blocks 11–12 removed and 24–29 duplicated) prove there
  is no tensor-mapping defect anywhere in the pipeline.
- The historical seams were never identity: LayerNorm followed by
  a covariance-colored or Xavier projection, twice in the
  residual stream.

**Workflow lessons (recorded in `run_registry.json` attempts):**
three diagnose attempts failed only at the evidence-posting step
(missing `GH_TOKEN`; then `issues: write` where the project's
proven configuration is `pull-requests: write`); the computations
themselves were unaffected. Both C14 workflows now match the
Campaign 11–13 configuration and include the partial-metadata
fallback publisher.
