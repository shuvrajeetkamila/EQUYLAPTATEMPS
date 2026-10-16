# Campaign 15 — Final Report: Healthy Connector Start Under Training

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Outcome (mechanical, from the frozen rules): TRAINING_DEPENDENT**

## Answers to the required questions

1. **Did native reconstruction still reproduce the recipient?**
   Yes — exactly. The fresh regression run on Campaign 15's code
   state reproduces Campaign 14: native repeat difference 0.0,
   reconstruction logit difference 0.0, all tensors identical
   (Gate A PASS).
2. **Is the new connector a true pass-through at initialization?**
   Yes — verified in code (bitwise identity) and behaviorally:
   every new-condition run starts at exactly 12.0150, the
   placement-only damage level, at all three seeds.
3. **Placement damage independent of connectors:** +9.24 NLL
   (34-block rearranged topology), paid identically by both
   conditions; it dwarfs everything either connector does after
   step ~16.
4. **Step-0 scores:** old 14.2481 / 13.2618 / 14.3053 vs new
   12.0150 (all seeds) — the new connector ahead by +1.25…+2.29.
5. **Step-32 scores (primary endpoint):** old 7.4577 / 7.3904 /
   7.4207 vs new 8.4624 / 8.2445 / 8.5319.
6–7. **Winner per seed at 32:** the **old** connector, at all
   three seeds (deltas −1.00, −0.85, −1.11; mean −0.99). Fully
   consistent — the Campaign 14 inversion is real, not noise.
8. **Steps 64/96:** the lead narrows (−0.15 at 64) and reverses
   at 96: new connector ahead at all three seeds (+0.25, +0.11,
   +0.11; mean +0.16, paired interval [−0.04, +0.36] — direction
   consistent, magnitude small).
9. **Did the initial advantage persist?** It disappeared during
   early training, then re-emerged by step 96. Hence the outcome
   label: the answer depends on training duration.
10. **Invalid/incomplete runs:** none. All six primary runs
    completed 96 steps with frozen parents, verified pairing
    (identical data order across conditions at every seed), and
    equal trainable counts (667,008). Two infrastructure attempts
    in earlier campaigns' style (skipped trigger evaluations) are
    recorded; no completed run was rerun.
11. **Gates:** A PASS (exact), B PASS, C FAIL (old connector
    consistently better at step 32), D: duration-dependent
    pattern confirmed (sign flip between 32 and 96). Held-out:
    `NOT_RUN_NO_PROMOTION`.
12. **What this establishes:** connector-inflicted damage is
    eliminated by the repaired connector at any seed; the
    historical 32-step comparison systematically favors the
    damaged start; training budget determines the connector
    ranking; placement damage (+9.24) remains the dominant cost.
13. **What it does not establish:** anything about heterogeneous
    transfer (untested here), any connector's superiority at a
    fixed budget in general, or a good absolute model (best
    child 6.78 vs native 2.77).
14. **Readiness:** not ready for a cross-family claim on the
    current placement/budget regime; ready for a focused
    optimization-dynamics study (see readiness memo).
15. **Single most defensible next step:** a preregistered
    trajectory study of connector learning dynamics at the
    self-control (learning-rate × budget interaction with the two
    inits, reporting full curves), because the connector question
    is now provably budget-relative — before any new donor work.

## Compact results (macro dev NLL, lower better; native = 2.7731)

| Seed | Old @0 | New @0 | Old @32 | New @32 | Old @96 | New @96 |
|---|---|---|---|---|---|---|
| 42 | 14.2481 | 12.0150 | 7.4577 | 8.4624 | 7.0311 | 6.7802 |
| 123 | 13.2618 | 12.0150 | 7.3904 | 8.2445 | 6.9079 | 6.7953 |
| 31415 | 14.3053 | 12.0150 | 7.4207 | 8.5319 | 6.9362 | 6.8271 |

Evidence: `paired_analysis.json`, `results_by_seed.csv`,
`checkpoint_results.csv`, `results_raw.jsonl`; jobs 38865626143 /
38865628701 / 38865631168 (SHA-verified bundles imported).
