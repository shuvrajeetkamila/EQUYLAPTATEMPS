# Campaign 10 Preregistration — The Awake-Control Experiment (FROZEN)

**Project:** HETEROGENEOUS AI MODEL SURGERY · **Frozen:** 2026-10-14, before
any Campaign 10 training. Pairing/sector/data/protocol/metric/gates identical
to Campaigns 7–9 (Mamba `1e76775…` × SmolLM `1d461723…`, sector [24,30),
macro dev NLL). Audit: `repository_audit.md`; awake rule:
`awake_control_design.md`.

## Hypothesis H10

Campaign 9's donor-specific advantage survives (a) a random twin explicitly
calibrated to remain **active** in donor-native geometry and (b) a generic
**trained output pathway** (no geometry preconditioning on the output seam)
shared identically by all children — and absolute quality improves toward
the gates. H0: the advantage was an artifact of twin inactivity and/or the
geometry-conditioned output; it vanishes against the awake twin.

## Frozen conditions

| # | Condition | Sector | Input init | Output init | Status |
|---|---|---|---|---|---|
| A | Campaign 9 natgeom (reference) | pretrained/twin | donor_native_geometry | donor_native_geometry | stored C9 records (historical, labeled) |
| B | Trained-output pretrained | pretrained | donor_native_geometry (input seam only) | generic `DenseSeam` default, trained | **new** |
| C | Trained-output raw twin | random (1234) | same | same | **new** |
| D | Trained-output **awake** twin | random (1234) + awake calibration | same | same | **new** |

Primary comparison: **B vs D**. Secondary: B vs C (what awake calibration
changes), and B's absolute quality vs Campaign 9's pretrained children.
Budgets: 32 steps primary (seed 42), seed-123 replication, then 96 steps —
registered by amendment before each stage, replication regardless of
direction. Trainable sets identical (888,768 seam params; awake calibration
adds no parameters and no training).

## Frozen gates (canonical, unchanged)

- **G-B:** benefit vs the matched twin (primary: D) ≥ 2% at seed 42 **and**
  ≥ 1% at seed 123. Benefit = 100·(L_twin − L_pre)/L_twin.
- **G-C:** language ratio ≤ 1.50 × host (4.5595) **and** macro ≤ 6.00.
- **Promotion:** all gates + causal ablation support + integrity; only then
  held-out opens once. Else `NOT_RUN_NO_PROMOTION`.

## Frozen outcome mapping (exactly one)

- **A:** B beats D by the gates, replicates, and meets G-C (held-out confirms).
- **B:** reproducible donor-specific advantage vs the awake twin, usefulness fails.
- **C:** valid experiment; advantage vs the awake twin absent/not replicated,
  or usefulness and advantage both absent.
- **D:** inconclusive — awake calibration fails its frozen success criterion
  or another documented validity failure prevents the decisive comparison.

Failed runs preserved; amendments dated and pre-filed; TEST never used.
