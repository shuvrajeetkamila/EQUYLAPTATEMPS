# Campaign 16 — Preregistration (frozen before any Campaign 16 run)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-15
**Type:** optimization-dynamics investigation on the same-family
self-control. No donor transfer; no held-out data.

## Primary question

Does a preregistered learning-rate sweep {9e-5, 3e-4, 9e-4}
(0.3×/1×/3× of the verified Campaign 15 baseline 3e-4) change the
old-vs-new connector ranking at the Campaign 13 placement, and do
gradient/update diagnostics explain the trajectory pattern?

## Matrix (18 primary runs)

Conditions A-old (`layernorm_linear` + `natgeom_in_trained_out`)
and B-new (`residual_identity`), seeds {42, 123, 31415}, each LR,
96 steps, validation at steps 0/8/16/32/64/96 via the
segment-driven hash-verified resume (Campaign 15 method).
Everything except connector and LR is identical to Campaign 15
(recipient in both roles, placement [11,13)←[24,30), data seed
42, AdamW wd 0, clip 1.0, batch 1, float32, 2 threads).
Diagnostics added (detached only): per-checkpoint parameter-update
norms per seam, fraction of parameters changed, graft activity
(input/output RMS, relative update), per-step gradient norms
(already in the library trace). Primary endpoint: step 32
(continuity with Campaign 15); trajectory is secondary.

## Gates (frozen)

- **Gate A:** fresh Campaign 14/15 diagnostics regression on the
  current code state must meet Campaign 14 tolerances (native
  repeat ≤ 0.001; reconstruction macro ≤ 0.01, max|Δlogits|
  ≤ 5e-3, tensors identical). Failure stops the panel.
- **Gate B (integrity):** every primary record shows COMPLETED
  96 steps, parents frozen, 667,008 trainable parameters, and
  per-seed data-order traces identical across the six
  connector×LR cells sharing the seed.
- **Gate C (reproducibility):** 18/18 runs accounted for;
  baseline-LR cells reproduce Campaign 15's scores within
  |Δmacro| ≤ 0.01 at every checkpoint (same seeds/segments);
  discrepancies investigated, not assumed harmless.
- **Gate D (LR interaction):** the frozen ranking rule (see
  `decision_rules.md`) applied per LR level.
- **Gate E (mechanism):** the frozen gradient signature (see
  `decision_rules.md`); supports or fails to support the
  LR-scale explanation; never upgraded beyond its wording.
- **Gate F (readiness):** recommendation mapping only.

Statuses, failure classification, and no-rerun rules as in
Campaign 15. Held-out: `NOT_RUN_NO_PROMOTION`.
