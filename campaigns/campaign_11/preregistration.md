# Campaign 11 Preregistration — The Learned-Structure Test (FROZEN)

**Project:** HETEROGENEOUS AI MODEL SURGERY · **Frozen:** 2026-10-14, before
any Campaign 11 training. Pairing/sector/protocol/metric/gates identical to
Campaigns 9–10 (Mamba `1e76775…` × SmolLM `1d461723…`, sector [24,30),
interface `natgeom_in_trained_out`, macro dev NLL). Audit:
`repository_audit.md`; control: `control_design.md`.

## Hypothesis H11

The intact pretrained sector outperforms a **statistics-preserved,
structure-destroyed** version of itself (within-tensor permutation of its
42 weight matrices, seed 20241014) under identical treatment — i.e., the
donor-specific effects of Campaigns 9–10 are attributable to *learned
organization* of the weights, not merely to their value statistics.
Alternatives (scale/statistics artifacts, control handicaps) are exactly
what this control removes. H0: the shuffled control matches or beats the
intact donor.

## Frozen conditions

| # | Condition | Sector | Status |
|---|---|---|---|
| A | Intact pretrained (Campaign 10 condition B) | original weights | stored C10 records (historical, labeled): 8.3665 / 8.3132 / 7.8392 |
| B | **Structure-destroyed** | within-tensor permuted (seed 20241014) | **new**: seeds 42, 123 @32; seed 42 @96 |
| C | Conventional random twin (Campaign 10 condition C) | native-init random | stored C10 records (reference) |
| D | Bypass | — | ablations inside every A/B run |

Primary comparison: **A vs B** at 32 steps, seeds 42 and 123; budget
comparison at 96 steps (seed 42). Benefit = 100·(L_B − L_A)/L_B.

## Frozen gates (canonical, unchanged)

- **G-B:** benefit vs B-control ≥ 2% (seed 42) and ≥ 1% (seed 123).
- **G-C:** language ratio ≤ 1.50 × host (4.5595) and macro ≤ 6.00 (on A).
- **Promotion:** all gates + control validity + causal ablation support +
  integrity → held-out opens once. Else `NOT_RUN_NO_PROMOTION`.

## Frozen outcome mapping (exactly one)

- **A:** replicated learned-structure advantage and usefulness gates pass.
- **B:** replicated learned-structure advantage; usefulness fails.
- **C:** valid control; no reliable learned-structure advantage.
- **D:** control invalid or other documented validity failure — inconclusive.

One permutation seed, fixed above; no permutation shopping. Failed runs
preserved; amendments dated and pre-filed; TEST never used.
