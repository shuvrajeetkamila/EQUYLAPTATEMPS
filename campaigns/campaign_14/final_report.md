# Campaign 14 — Final Report: Surgery Damage Elimination

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Outcome (mechanical, from the frozen preregistration):**
**RECONSTRUCTION_VALIDATED**

## The required statements

- **Exact reconstruction:** PASS, exactly. With the recipient's own
  sector reinserted at its native depth and seams bypassed (V1),
  the child is bit-identical to the untouched recipient: all 273
  audited tensors equal, maximum logit difference 0.0, score
  degradation 0.0000 (2.7731 = 2.7731).
- **Baseline reproducibility:** PASS, exactly. Two native
  evaluations agree to 0.0 (macro 2.7731 both times, identical
  per-example scores), reproducing Campaign 13's independent
  native measurement.
- **Largest damage operation:** the **standard connector seams**.
  Inserted at native placement with recipient weights otherwise
  untouched, they alone cost **+21.69 NLL** at initialization
  (2.77 → 24.47). The Campaign 13 placement (relocation +
  removal of blocks 11–12 + duplication of 24–29 + depth 30→34),
  with seams neutralized, costs **+9.24**. The historical
  configuration starts at **+11.47** (factors interact; the
  combined damage is less than the seams-alone figure because the
  placement changes what the seams see).
- **Connector identity-preserving:** the historical DenseSeam is
  not (LayerNorm + non-identity projection, twice). The new
  `ResidualDenseSeam` (zero-initialized residual branch) **is
  exactly identity at initialization**, proven offline
  (bit-equality inside a real child) and at real scale: V3 = V1 =
  recipient, degradation 0.0000.
- **Relocation/re-depth contribution:** with the connector
  neutralized, the Campaign 13 placement alone costs +9.24 NLL —
  it is the second-largest damage source and cannot be repaired
  by any connector; it can only be avoided (native placement) or
  compensated by training.
- **What was fixed:** (i) the connector design — an opt-in
  `residual_identity` interface architecture that is
  function-preserving at initialization; (ii) this campaign's own
  workflow defects (token/permissions), recorded in the registry.
  No historical code path or default changed.
- **What remains unresolved:** at the relocated placement, 32
  steps of seam training from the identity start (V7: 12.0150 →
  8.4623) recovered less than from the historical start (V8:
  14.2481 → 7.4578, replicating Campaign 13's F = 7.4577 to four
  decimals). Identity initialization removes the connector's own
  damage but confers no short-budget training advantage where
  reconstruction is impossible; whether it wins at larger budgets
  is untested and was not claimed.
- **Gates:** Gate 1 PASS (exact), Gate 2 PASS (exact), Gate 3
  PASS (exact). The trained stage ran legitimately after the
  gates. Held-out: `NOT_RUN_NO_PROMOTION` (not applicable here).

## Verdict on the Campaign 13 mystery

The 2.77 → 7.46 gap was overdetermined: catastrophic non-identity
seams (+21.7 alone), a placement that removes, duplicates, and
reorders blocks (+9.2 alone), and a 32-step seam-only training
budget that recovered the combined child only from 14.25 to 7.46.
There was never a tensor-mapping bug (273/273 and 309/309 tensors
bit-identical), an evaluation inconsistency (native 2.7731
reproduced exactly in three independent jobs), or a checkpoint
problem. **Donor transfer was never the bottleneck; the surgery
regime was** — and its single largest component was the connector
initialization, now repairable by construction.

## Whether further transfer experiments are justified

Conditionally yes — see the decision memo. The pipeline now
provably preserves the recipient's own behavior when the surgery
is function-preserving by construction (native placement +
residual-identity seams). A future heterogeneous campaign may
resume only under the conditions listed there; Campaign 15 is
not started by this report.
