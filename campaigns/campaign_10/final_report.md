# HETEROGENEOUS AI MODEL SURGERY — Campaign 10 Final Report

**Campaign 10: The Awake-Control Experiment** · Status: **COMPLETE** ·
Development-only; held-out **NOT_RUN_NO_PROMOTION** · Branch
`arena/97efdf4a-equylaptatemps` (session-fixed), PR #1.

## Final decision: Outcome B — donor-specific effect without useful inheritance (fragile, control-dependent)

Campaign 10 asked whether Campaign 9's advantage was an artifact of a
sleeping twin. Answer, in one sentence: **the advantage over the awake twin
is real at 32 steps and replicates, but it is small against the raw twin,
shrinks with training, and reverses against the raw twin at 96 steps —
while absolute quality, though the best yet for a pretrained child in this
pairing, remains far from useful.**

| Steps | Seed | B pretrained | C raw twin | D awake twin | Benefit vs D | Benefit vs C | G-C |
|---|---|---|---|---|---|---|---|
| 32 | 42 | 8.3665 | 8.8099 | 9.3403 | **+10.43%** | +5.03% | FAIL |
| 32 | 123 | 8.3132 | 8.3945 | 8.6976 | **+4.42%** | +0.97% | FAIL |
| 96 | 42 | 7.8392 | 7.7088 | 7.9318 | +1.17% | **−1.69%** | FAIL |

(Pretrained language ratios 1.87–1.92× at 32 steps, 1.835× at 96; ceilings
1.50× and macro 6.00. Historical reference A — Campaign 9's natgeom
pretrained 9.0174/8.6500/7.8413 — is improved on by B at matched budgets at
32 steps: the trained output pathway did help absolute quality modestly.)

## Reading the three controls honestly

- **Vs D (awake twin, the preregistered primary):** G-B's frozen thresholds
  pass at both 32-step seeds. But D's calibrated activity is destructive
  (amplified random transformations damage the child; D is the worst child
  in every row), so beating D is necessary-but-weak evidence.
- **Vs C (raw/sleeping twin):** +5.03% and +0.97% at 32 steps — the most
  conservative estimate of the donor-specific effect: small, and at seed
  123 within the range this programme has learned to treat as noise-prone.
- **At 96 steps:** B beats D by only +1.17% and *loses* to C (−1.69%).
  Whatever donor-specific signal exists at short budgets, longer training
  lets the twins' interfaces compensate — the effect is budget-fragile.

Synthesis across Campaigns 9–10: donor-specific advantages in this pairing
appear precisely when the control is disadvantaged (geometry silences the
twin; awake scaling poisons it), and hover near zero against the strongest
control at longer budgets. The pretrained sector's contribution is real
(causal ablations: B bypass 12.68/10.61/9.35 vs intact; zero 11.16–21.10)
but small and not the bottleneck to usefulness — absolute quality is.

## Answers to the brief (condensed)

1–2. Question H10; Campaign 9 audited and recomputed exactly
(`repository_audit.md`). 3. Revisions/sector frozen from C9. 4. Interface:
geometry input + trained output; awake rule frozen in
`awake_control_design.md` (target 1.0282 measured natively). 5–6.
Conditions B/C/D (A historical), seeds 42/123, budgets 32/96 — all
completed; exploratory screen D preserved and labeled. 7–9. Full results
and ablations in the table and `ablation_results.json`; awake twin
demonstrably active (update ≈0.60 vs raw ≈0.001). 10. Trainable 888,768 in
all conditions (parity verified). 11. Cost ≈5,400 s CPU across five jobs;
no GPU. 12. Failures: none scientific; one implementation deviation
(bound clipping) corrected by Amendment 01 before confirmatory runs.
13. Deviations: Amendment 01 only, pre-filed. 14. Bundles SHA-256-verified
(hashes in `results.json`). 15. Tests/CI: `test_report.txt`, PR #1.
16. Gates: G-B (vs D) passes at 32 steps ×2 seeds; G-C fails everywhere.
17. Held-out untouched. 18. Classification: **Outcome B** (the frozen
definition — replicated advantage vs the preregistered primary control,
usefulness unmet — applied without post-hoc control swapping; the
fragility above is the headline caveat, not a footnote). 19. Strongest
justified conclusion: pretrained SmolLM blocks contribute a small,
causally real, donor-specific signal under geometry-matched interfaces at
short budgets; it is neither stable across budgets nor close to useful.
20. Limitations: two seeds; awake activity is destructive rather than
benign; single sector/pairing; activity target is second-order only.
21. Next: see decision memo — the decisive remaining question is whether
*any* active-and-benign control can be constructed (e.g., a random sector
trained briefly to be benignly active, or an activity-matched *pretrained-shuffled* control); if the effect cannot survive that, the
programme's donor-specific signal should be considered an interface
artifact family, not transferable capability.

## Verification

Preregistration frozen at `52e8598` before training; Amendment 01 pre-filed
before confirmatory runs; all numbers regenerated from SHA-256-verified
records by `scripts/publish_campaign10.py`, traced by
`tests/test_campaign10_results.py`.
