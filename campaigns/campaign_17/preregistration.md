# Campaign 17 — Preregistration (frozen before primary results)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-16
**Type:** native-depth placement redesign (self-control only).
No foreign donor, no training, no held-out data.

## Primary question

Can the pipeline preserve native depth (30 blocks), original
order outside a declared region, and all unrelated parameters —
and what damage, if any, do (a) the replacement interface and
(b) a matched-depth structural exchange cause?

## Modes (one diagnostics job; definitions in
`implementation_design.md` and `native_depth_specification.md`)

- **A** native recipient, evaluated twice (reference).
- **B** exact reconstruction via the historical path
  ([24,30)←[24,30), seams bypassed) — Campaign 14 continuity.
- **C** identity replacement via the new engine: identity order,
  graft region [11,13) supplied by the donor-role copy of the
  same checkpoint, residual-identity seams (the normal interface).
- **D1** matched-depth exchange: swap sectors [11,13) ↔ [24,26),
  seams bypassed. Structural intervention; damage is measured,
  not gated to be small.
- **D2** adjacent exchange: swap blocks 15 ↔ 16, seams bypassed
  (secondary; distinct locality question).

Evaluation identical to Campaigns 14–16 (dev macro NLL; fixed
4-example logit batch; per-example scores; graft activity).

## Frozen criteria (Campaign 14/16 tolerances, unrelaxed)

- Gate A (Mode B): macro |Δ| ≤ 0.01 vs native, max|Δlogits|
  ≤ 2e-2 and mean ≤ 1e-3 (Amendment-01 form), all tensors
  identical, native repeat |Δ| ≤ 0.001.
- Gate B (Mode C): same score/output criteria; manifest exact
  (identity permutation, graft source = donor-role copy);
  trainables confined to seams.
- Gate C (structural): every mode's manifest verifies against
  measured block fingerprints: 30 blocks, no duplicates, no
  removals, deviations exactly as declared.
- Gate D: exchanges executed exactly as registered (manifests);
  damage reported. "Substantial" = degradation > 1.0 NLL.
- Gate E: records complete and internally consistent.
- Gate F → outcome label per `decision_rules.md`.

If Gate A or B fails, downstream modes are still reported but
the campaign outcome is PLACEMENT_IMPLEMENTATION_FAILURE and no
readiness claim is made. Held-out: `NOT_RUN_NO_PROMOTION`.
