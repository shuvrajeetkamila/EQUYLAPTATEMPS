# Campaign 18 — Decision rules (frozen)

## Gates

- **G1 Measurement**: native repeat ≤ 0.001; reconstruction and both
  identity controls exact under frozen tolerances (macro |Δ| ≤ 0.01,
  logit max ≤ 0.02 / mean ≤ 0.001, tensors unchanged); all five
  invalid-manifest probes rejected; parents unchanged.
- **G2 Map completeness/consistency**: ≥ 90% of development
  conditions produce valid scores; replication anchors agree with
  Campaign 17 (val-xchg-a11-c24-L2 damage within 0.05 of +8.3953;
  val-swap-15 within 0.05 of +0.1547).
- **G3 Predictor**: chosen predictor meets the frozen held-out
  criterion (held-out MAE ≤ 0.8 × constant-baseline held-out MAE).
- **G4 Exact-location replacement structurally verified** (Campaign 17
  Mode C evidence + Campaign 18 engine controls).
- **G5 Pilot design completeness**: the Campaign 19 draft names one
  donor (exact revision), recipient, sector, location, controls,
  evaluation, seeds, and a decision rule that does not depend on
  post-hoc selection.
- **G6 Authorization**: an explicit user decision to run a foreign
  pilot. Not granted within Campaign 18's brief; therefore
  `CAMPAIGN_19_PILOT_AUTHORIZED` is unreachable in this campaign.

## Label precedence (first match wins)

1. `MEASUREMENT_SYSTEM_INVALID` — G1 fails.
2. `STRUCTURAL_DAMAGE_MAP_INCONCLUSIVE` — G2 fails.
3. `PREDICTIVE_MAP_FAILED_HELDOUT` — G3 fails.
4. `CAMPAIGN_19_PILOT_AUTHORIZED` — G1–G6 all pass (unreachable here).
5. `PLACEMENT_AWARE_GRAFT_DESIGN_READY` — G1–G5 pass.
6. `PLACEMENT_AWARE_GRAFT_DESIGN_BLOCKED` — G1–G4 pass, G5 fails.
7. `STRUCTURAL_DAMAGE_MAP_VALIDATED` — otherwise (map valid;
   predictor outcome already handled by rule 3 if it failed).
