# Campaign 16 — Final Report: Optimization-Dynamics Investigation

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Outcome (mechanical, frozen rules): LEARNING_RATE_SENSITIVE**
(with the preregistered Gate E mechanism signature supported)

## Answers to the required questions

1. **Native reconstruction regression:** PASS under Amendment 01
   (attempt 1 recorded FAILED on the frozen 5e-3 logit proxy —
   cross-CPU float noise with macro difference exactly 0.0 and
   273/273 tensors identical; the amended proxy bound and the
   failure are preserved in the record).
2. **All 18 primary runs complete:** yes — on attempt 2. Attempt 1
   also completed all training, but its score evidence was
   destroyed by the posting channel (F-C16-002); attempt 2 uses
   the identical frozen design with compacted records.
3. **Step-32 scores by connector and learning rate** (macro dev
   NLL, lower better; seeds 42/123/31415):
   - LR 9e-5: old 7.93/7.81/7.90 vs new 10.48/10.40/10.52
   - LR 3e-4: old 7.46/7.39/7.42 vs new 8.46/8.24/8.53
   - LR 9e-4: old 7.34/7.22/7.26 vs new 7.10/7.18/7.03
4. **Paired differences (old − new) at step 32:** −2.59 (low),
   −0.99 (base), +0.17 (high) on average; every seed agrees in
   sign within each level.
5. **Does ranking change with learning rate?** Yes, monotonically:
   the new connector's relative position improves at every
   checkpoint as LR rises (step-96 means: −1.04 → +0.16 → +0.91).
   At 3× LR the new connector produces the best children overall
   (≈6.04–6.11 at step 96).
6. **Does ranking change with duration?** At low/base LR yes
   (C15's pattern); at high LR the new connector leads from step
   32 onward — duration-dependence itself is LR-dependent.
7. **Gradient/update measurements:** at baseline LR the new
   connector's median steps 1–32 gradient norms are 0.75–0.93×
   the old connector's (same direction at all seeds; median
   0.77, outside the frozen [0.8, 1.25] band); clipping is active
   on 100% of steps for both; by step 96 the new connector's
   seams have moved *further* from initialization (update L2
   ≈3.8/5.9 vs ≈2.0/2.1) — it starts at exactly zero and grows.
8. **Learning-rate explanation:** supported as a pattern (frozen
   Gate E signature holds: gradient gap + predicted LR response
   +2.75 NLL swing). Per the frozen caveat: consistent with LR
   sensitivity; not proof that LR is the sole cause — the
   zero-init geometry and the LR scale are two descriptions of
   the same mechanism.
9. **Placement harm:** unchanged and dominant — every child at
   every setting remains ≥3.3 NLL worse than the native 2.7731
   (best child 6.04); placement-only damage is +9.24.
10. **Gates:** A PASS (amended; attempt-1 failure preserved),
    B PASS (integrity/pairing verified), C PASS (baseline cells
    reproduce Campaign 15 within 7.2e-05), D: LR interaction
    confirmed, E: mechanism signature supported, F: see memo.
11. **Invalid/incomplete/rerun runs:** attempt 1 (18 runs)
    completed training but lost score evidence to the posting
    channel (F-C16-002, documented; salvaged registries retained,
    scores unused); attempt 2 is the analysis source. No
    completed-and-recorded run was ever rerun.
12. **What this establishes:** the Campaign 15 reversal is
    primarily a training-scale phenomenon: at the shared LR the
    zero-start connector is effectively under-trained early; the
    old connector's step-32 "advantage" is an artifact of
    comparing unequally-scaled optimization, and it inverts with
    either more steps or more learning rate.
13. **What it does not establish:** anything about foreign-donor
    transfer; an optimal LR (three points only); superiority of
    either connector as a surgery (both remain far from native
    under this placement).
14. **Readiness:** ready for a placement redesign study, not yet
    for a cross-family claim (see `campaign_17_readiness.md`).
15. **Single most defensible next step:** design and gate a
    native-depth (function-preserving) placement for the
    self-control — the largest remaining damage source by an
    order of magnitude — before any further connector or donor
    work.

Evidence: jobs 38882548685 / 38885462762 / 38888327387
(SHA-verified); `paired_analysis.json`,
`learning_rate_sensitivity.json`, `optimization_diagnostics.csv`,
`results_by_seed.csv`. Held-out: `NOT_RUN_NO_PROMOTION`.
