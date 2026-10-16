# Campaign 18 — Preregistration: Structural Damage Mapping and Placement-Aware Graft Design

Frozen before any Campaign 18 primary outcome. Anchors: Campaign 17
(`dc386bc`), native macro 2.773126 (run 38899506528, re-verified in
`repository_audit.md`).

## Question

Within the frozen recipient (SmolLM-135M, 30 blocks), how does the
functional damage of moving blocks depend on displacement distance,
source/destination position, direction, number of blocks displaced,
and section length — and can damage for *unseen* placements be
predicted well enough, under a frozen rule, to inform a future
placement-aware foreign-graft design?

Primary outcome: `damage = condition_macro − native_macro` on the
frozen Campaign 14–17 dev panel (macro dev NLL; absolute differences
only; relative change reported separately where noted).

## Conventions (frozen)

- `execution_order` lists block identities in execution order.
  `displacement(block) = final_position − original_index`.
- Adjacent swap at `p`: identities at positions p, p+1 exchange.
- Single-block move `(b, d)`: block b is removed and reinserted so
  its final position is `b + d`; intervening blocks shift by one.
- Section exchange `(a, c, L)`: sections `[a, a+L)` and `[c, c+L)`
  swap places, internal order preserved; intervening blocks shift.
  Directly exchanged and indirectly shifted blocks are recorded
  separately via the displacement map.

## Placement sets (frozen; `config/placement_sets.json`)

Generation seed 1800; sets SHA-256 `bcd997475652bb03…` (full hash in
`config/preregistration.json`). Development 48 (even adjacent swaps,
moves from blocks {2,8,14,20,26}, six section exchanges); validation
5 (swaps 7/15/21 — 15 replicates Campaign 17 D2 — and two exchanges,
one replicating Campaign 17 D1); held-out 30 (remaining odd swaps,
moves from blocks {5,11,17,23}, three exchanges). Held-out outcomes
are not inspected until the predictor is frozen and the opening is
recorded in `heldout_access_log.json`.

## Measurement validation (Stage 2; gates everything)

Native ×3; historical reconstruction; engine identity with residual
seams; engine no-op (identity order, bypass seams, no donor);
inverse-operation restoration for three development conditions;
five invalid-manifest probes (duplicate identity, missing block,
unknown identity, non-permutation order, declared-order mismatch)
that must all be rejected; recipient fingerprint before/after.
Frozen tolerances (from Campaigns 14–17): macro exactness |Δ| ≤
0.01, logit max ≤ 0.02 / mean ≤ 0.001, native repeat ≤ 0.001, all
audited tensors identical for identity-class conditions.

## Predictor (frozen candidates; fit on development only)

Features per condition: `max_abs_displacement`,
`n_blocks_displaced`, `mean_source_position_norm` (mean original
index of displaced blocks / 29).

- P0: constant = development mean damage.
- P1: OLS `damage ~ 1 + max_abs_displacement`.
- P2: OLS `damage ~ 1 + max_abs_displacement + n_blocks_displaced +
  mean_source_position_norm`.

Selection: lowest MAE on the 5 validation conditions; ties broken
toward the simpler model (P0 < P1 < P2). Coefficients are then frozen
(development fit only) and the held-out set is evaluated **once**.
Held-out success (frozen): chosen predictor's held-out MAE ≤ 0.8 ×
P0's held-out MAE. No refitting against held-out outcomes.

## Mechanism diagnostics (Stage 6, bounded)

For native + four selected conditions, position-wise hidden-state
RMS ratio and cosine similarity versus native at each executed
block's output (diagnostic batch of 4). Descriptive only.

## Statistics

Evaluation is deterministic; "seeds" enter through the frozen
placement generation (seed 1800). Uncertainty: repeat-measurement
spread (native ×3, expected 0) and per-example NLL distributions
(bootstrap CIs computed publisher-side from recorded per-example
rows). Repeated conditions are not independent model replications
and are never described as such.

## Exclusions, failures, amendments

A condition whose child fails to build or score is recorded as a
failure with its stage, never silently dropped or converted to zero.
>10% missing development conditions ⇒ map incomplete (Gate B fails).
Amendments follow the standing format (original rule, replacement,
reason, whether primary outcomes were seen, timestamp, commit).

## What Campaign 18 will not do

No foreign donor in any scored condition; no training; no held-out
capability evaluation (`NOT_RUN_NO_PROMOTION` stands); no Campaign 19
execution. Campaign 19 authorization cannot be self-granted (see
`decision_rules.md`, Gate G6).
