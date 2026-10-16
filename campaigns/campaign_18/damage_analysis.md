# Campaign 18 — Damage analysis (Stage 4)

Evidence: 83 scored placements (48 development, 5 validation, 30
held-out), deterministic scoring on the frozen panel; native
2.773126 (repeated ×3 exactly). Numbers from
`structural_damage_map.csv`, `position_effects.csv`,
`distance_effects.csv`, `interaction_analysis.json`,
`damage_model.json`. Bootstrap CIs (400 resamples over per-example
damages) accompany every condition in the map.

## What predicts damage

1. **Position dominates for adjacent swaps.** Interior swaps cost
   +0.03…+0.33 (mean ≈ +0.10), but swapping blocks 0↔1 costs +6.36
   and 28↔29 costs +10.59. "Adjacent" is not a safety class; the
   first and last blocks are special.
2. **Direction matters.** Moving a block toward the input (earlier)
   is consistently worse than toward the output at equal distance
   (block 8 at distance 8: +7.42 earlier vs +0.22 later; block 14:
   +2.92 vs +0.44; block 26: +2.74 vs later moves ≤ +0.44). See the
   direction×distance table in `interaction_analysis.json`.
3. **Distance matters, but non-monotonically and interactively.**
   Long moves by late blocks can be mild (block 20, ±8: +0.35/+0.14;
   block 2 later +8: +0.17) while the same distance early is severe
   (block 8 earlier −8: +7.42). Equal distances are not equal
   interventions — the Campaign 17 distance-only intuition is a
   first approximation, not a law.
4. **Section exchanges are the harshest class** (+3.25 for L3
   [12,21]↔…, +7.3–+8.4 for the L2 distant exchanges), consistent
   with displaced-block count and boundary disruption combining.
5. The position-aware predictor (P2) generalized: held-out MAE
   0.852 vs 1.648 for the constant baseline (frozen criterion ≤0.8×
   met at 0.52×). It is useful for ranking placements, not precise
   in absolute terms (worst held-out misses in `damage_model.json`).

## Contradictions and cautions

- Block 2 moved one step *earlier* costs +0.92 while many longer
  later moves cost < +0.2 — local exceptions are real.
- No condition improved on native beyond noise; there is no
  "beneficial rearrangement" finding.
- One checkpoint, one panel: the map is a SmolLM-135M map, not a
  universal transformer law (see
  `architecture_mechanism_analysis.md`).
