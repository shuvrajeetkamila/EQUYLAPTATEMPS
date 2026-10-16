# Campaign 18 — Final report: Structural Damage Mapping and Placement-Aware Graft Design

**Primary label: `PLACEMENT_AWARE_GRAFT_DESIGN_READY`.**
Held-out capability evaluation: `NOT_RUN_NO_PROMOTION` (unchanged).
Foreign conditions run: 0. Training runs: 0.
Evidence: Actions runs `38907871417` (map) and `38911632615`
(held-out), SHA-verified bundles in `map_38907871417/` and
`heldout_38911632615/`.

## What was done

Stages 0–9 complete; Stage 10 (this decision) applies the frozen
rules. Verified Campaign 17 from raw records (no discrepancies),
froze 83 placements (48 development / 5 validation / 30 sealed
held-out, SHA-256 `bcd997475652bb03…`), validated the measurement
system, mapped damage, froze a predictor on development+validation,
and confirmed it in a single recorded held-out opening.

## Headline findings

| Condition class | n | Damage vs native 2.773126 |
|---|---|---|
| Interior adjacent swaps | 27 | ≈ +0.03…+0.33 (mild) |
| Boundary swaps 0↔1 / 28↔29 | 2 | +6.36 / +10.59 (severe) |
| Single-block moves toward input | 13 | up to +7.42 (block 8, −8) |
| Single-block moves toward output | 13 | ≤ +0.44 at distance 8 |
| Section exchanges | 11 | +3.25…+8.40 (harshest class) |
| Identity/reconstruction controls | 4 | 0.000000 (bitwise, repeated) |

- **Position beats distance.** The first and last blocks are
  load-bearing; interior neighbors are nearly interchangeable.
- **Direction is asymmetric.** Moving a block earlier (toward the
  input) is systematically worse than later at equal distance.
- **Distance alone is not a law.** Late blocks tolerate long moves
  (block 20 ±8: ≤ +0.35) that are catastrophic early (block 8 −8:
  +7.42). Interactions are documented in `interaction_analysis.json`.
- **Predictor generalizes.** Frozen position-aware model (P2):
  held-out MAE 0.852 vs 1.648 constant baseline (criterion ≤ 0.8×;
  achieved 0.52×) — good enough to rank unseen placements, not to
  quote exact scores (`damage_model.json` lists its worst misses).

## Gates

G1 measurement PASS (controls bitwise; all five invalid-manifest
probes rejected; parents unchanged) · G2 map PASS (83/83 scored;
Campaign 17 anchors replicated exactly: +8.3953, +0.1547) · G3
predictor PASS · G4 exact-location replacement PASS (Campaign 17/18
controls) · G5 design complete PASS · G6 user authorization NOT
GRANTED (by design — this campaign cannot self-authorize).

## What this does and does not establish

Established: structural damage in this recipient is measurable,
patterned (position, direction, displaced-block count), and
predictably rankable on unseen placements. Not established: that
any foreign component will work at any location — low native
movement damage is not foreign compatibility, and Campaign 18 ran
no foreign weights. The transplant hypothesis remains unproved.

## Most defensible next step

A user decision on the Campaign 19 draft (recipient sector [11,13),
Mamba donor under declared residual connectors, scrambled-donor and
identity controls, Campaign 16's frozen LR). If declined, the map
itself stands as the campaign's contribution.

Full required answers (publication state, exact SHAs, CI) are in
`publication_verification.md`; reasoning in `decision_memo.md`.
