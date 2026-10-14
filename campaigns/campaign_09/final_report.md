# HETEROGENEOUS AI MODEL SURGERY — Campaign 9 Final Report

**Campaign 9: Donor-Native Representation Geometry** · Status: **COMPLETE** ·
Development-only; held-out **NOT_RUN_NO_PROMOTION** · Branch
`arena/97efdf4a-equylaptatemps` (session-fixed), PR #1.

## Final decision: Outcome B — donor-specific improvement without useful absolute performance

For the first time since Campaign 4, a pretrained donor sector beat its
matched random twin — and this time the effect is large and replicated.
Initializing the interfaces to reconstruct the donor's native representation
geometry (whiten the child stream, then color it with SmolLM's own native
covariance) produced:

| Condition | Steps | Seed | Pretrained | Twin | Benefit | G-B | G-C |
|---|---|---|---|---|---|---|---|
| original | 32 | 42 | 8.5719 | 7.5349 | −13.76% | FAIL | FAIL |
| original | 32 | 123 | 9.4023 | 7.8900 | −19.17% | FAIL | FAIL |
| **natgeom** | 32 | 42 | 9.0174 | 9.9632 | **+9.49%** | PASS | FAIL |
| **natgeom** | 32 | 123 | 8.6500 | 9.6512 | **+10.37%** | PASS | FAIL |
| **natgeom** | 96 | 42 | 7.8413 | 8.8588 | **+11.49%** | PASS | FAIL |

Causal tests support donor-specificity: bypassing the sector hurts the
geometry children (10.31/9.62/9.88 vs intact 9.02/8.65/7.84), zeroing hurts
far more (18.78/24.76/18.44), and the pretrained graft's relative update
returns to native-like magnitudes (0.54–1.33, vs 19–58 under earlier
interfaces). But usefulness fails everywhere (language 1.80–1.96× host vs
1.50 ceiling; macro 7.84–9.02 vs 6.00), so there is no promotion and the
held-out set was never opened.

## The honest interpretation (read before celebrating)

Two things are true at once:

1. **The donor-specific effect is real** under the preregistered criterion:
   identical geometry treatment, identical everything, only the sector
   weights differ — and pretrained weights win by ~10% at two seeds and two
   budgets. Given donor-native geometry, pretrained blocks specifically
   contribute; random blocks cannot exploit that geometry. H9's core
   prediction is supported.
2. **The geometry interface mainly disables the twin rather than raising
   the pretrained child.** Under donor-native coloring the random sector
   becomes a near-identity pipe (relative update ≈0.001; intact ≈ bypass),
   so the twin approximates a bypass control — while the pretrained child's
   absolute score (8.65–9.02) is no better than under the original
   interface (8.57–9.40). "Reconstructing geometry unlocked latent donor
   value in absolute terms" is therefore **not** supported; what is
   supported is narrower: *the pretrained sector's contribution is
   specifically enabled by its native geometry, and is causally necessary
   within that condition.* This is also why the classification is B, not A,
   and why no inheritance claim is made.

## Answers to the brief's final-report questions (condensed)

1. Question: H9 (above). 2. Prior campaigns: dependence without benefit
   across interfaces/families/budgets; C4's small mandatory-pathway effect
   the only prior positive. 3. Hypothesis and 4. revisions/sector: frozen in
   `preregistration.md` (Mamba `1e76775…`, SmolLM `1d461723…`, sector
   [24,30)). 5. Interface change: §Implementation Audit. 6. Geometry
   measured from native donor/host streams on TRAIN text (covariance
   tensors; `representation_geometry.md`). 7–9. Conditions/seeds/budgets:
   table above (6 new runs; originals stored from C7). 10. Absolute
   quality: fails G-C everywhere. 11. Ablations: donor necessary within
   condition; interface-necessity shown by the original-interface rows.
   12. Stability: no non-finite metrics; parents unchanged in all jobs.
   13. Cost: 1,160 s + 717 s + 2,878 s CPU runner time; no GPU. 14. Failed
   runs: none in Campaign 9 (the frozen-C8-method guard failure belongs to
   Campaign 8's record). 15. Deviations: none beyond the pre-filed
   Amendment 01 (replication + budget registration). 16. Tests/CI: see
   `test_report.txt` and PR #1. 17. Held-out: untouched,
   `NOT_RUN_NO_PROMOTION`. 18. Classification: **Outcome B**. 19. Strongest
   justified conclusion: donor-native geometry is a real, causal moderator
   of donor-specific contribution — the first replicated one in the
   programme — but at tested budgets it does not produce useful children.
   20. Limitations: second-order geometry only; twin near-identity caveat
   (above); single donor sector and pairing. 21. Next experiment: see
   decision memo — combine geometry-matched input with a *trained* output
   pathway and an active-twin control (a random sector pre-scaled to be
   non-identity in donor geometry), to test whether the effect survives a
   stronger control and whether absolute quality can be recovered. Not
   launched.

## Verification

Preregistration frozen at commit `1648f47` before training; every number
regenerated from SHA-256-verified records by `scripts/publish_campaign09.py`
(bundle hashes in `results.json`) and traced by
`tests/test_campaign09_results.py`. Final remote SHA/CI: PR #1 and the
publication commit closing this campaign.
