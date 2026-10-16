# HETEROGENEOUS AI MODEL SURGERY — Campaign 12

## The Structure-Scale Test: Does learned weight arrangement matter at different levels of scrambling?

**Classification: Outcome C** — both scrambled controls were valid,
and the intact donor sector showed **no reliable advantage** at
either scrambling level. Promotion: no. Held-out:
`NOT_RUN_NO_PROMOTION`.

### The question

Campaign 11 scrambled every weight inside the donor sector's
matrices and found the intact donor's edge didn't replicate. A fair
objection remained: maybe that scramble was simply too violent.
Campaign 12 therefore tested two levels of scrambling — a gentler
one that shuffles weights only *within each row* (keeping every
row's statistics exactly), and Campaign 11's full scramble
(reproduced bit-for-bit, verified by cryptographic fingerprints) —
plus a brand-new random seed, fresh runs of every condition, and a
second training budget. 20 runs in total, all rules frozen in
advance.

### The answer, in numbers (macro development NLL, lower is better)

Primary panel, 32 training steps — intact donor's advantage over
each control (positive = intact better):

| Seed | vs within-row scramble | vs full scramble | vs random twin |
|---|---|---|---|
| 42 | +5.82% | +2.78% | +5.03% |
| 123 | +1.58% | −0.57% | +0.42% |
| 31415 (new) | −3.55% | −5.04% | +0.87% |

At 96 steps the pattern was likewise mixed-to-negative. The frozen
rule required the intact donor to beat *both* scrambles at *all*
seeds. It managed that at exactly one seed of three. The scrambled
sectors were proven active and healthy (the child leaned on them
as hard as on the real donor), so this null result cannot be
blamed on broken controls.

### What it means

Four campaigns (9–12) now converge: the surgical interface makes a
foreign model part *work* inside a different architecture, and
pretrained weight statistics give small, luck-of-the-seed benefits
— but there is no reproducible evidence that the donor's learned
internal arrangement, at either fine or coarse granularity, is
what helps. Absolute quality also remains well short of useful
(language quality ~1.9× the host against a 1.50× ceiling).

### Record

Preregistration frozen before training (commit `3f5df5f`);
validation job `38726596264`; run jobs `38726922262` and
`38738479882`, all SHA-256-verified on import. One disclosed
amendment corrects a publisher check to the exact frozen validity
wording (verdict unaffected). 266 tests pass; lint clean. Full
detail: `campaigns/campaign_12/final_report.md` and
`decision_memo.md`.
