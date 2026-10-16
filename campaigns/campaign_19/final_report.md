# Campaign 19 — Final report: First Controlled Foreign-Graft Pilot

**Frozen label: `DONOR_ADVANTAGE_NOT_SHOWN`.**
Reporting classification: `VALID_NEGATIVE_NO_DONOR_ADVANTAGE`.
Held-out: `NOT_RUN_NO_PROMOTION` (primary rule failed; never opened).
Evidence: gates run `38947055805`, training run `38950005531`
(SHA-verified bundles); failed attempt `38947619114` preserved
(F-C19-001, no scores).

## A. What actually ran

Stage 0–2 preflight and run-matrix reconciliation (10 training
runs: C/D/E × 3 seeds + H × 1); Stage 3 donor-validity gate;
Stage 4 identity gate; Stage 5 structural/gradient audit; Stage 6
the full frozen training matrix; Stage 7 controlled evaluation;
Stage 8 ablation/restoration; Stage 9 held-out decision (not
opened); Stages 10–12 analysis, tests, publication.

## B. Donor validity gate — PASS

Intact Mamba donor 3.5206 vs its own sector-scrambled self 3.7423
(margin +0.2216 ≥ frozen 0.05). The donor sector does carry learned
structure at home.

## C. Identity control — PASS (bitwise)

Identity replacement macro 2.773125 = native; logit differences 0.0.

## D–E. The graft result (macro dev NLL; native = 2.773126)

| Seed | Intact C (step 32) | Scrambled D | Random E | Advantage vs best control |
|---|---|---|---|---|
| 42 | 7.161 | 7.128 | 7.239 | **−0.033** |
| 123 | 7.240 | 7.163 | 7.335 | **−0.077** |
| 31415 | 7.162 | 7.085 | 7.436 | **−0.077** |

Frozen rule required mean advantage ≥ +0.25 with ≥ 2/3 seeds
positive. Observed: **−0.063, 0/3 positive.** The intact donor did
not beat its controls — it lost narrowly to the scrambled donor in
every seed (at step 96 likewise: scrambled 6.64–6.67, intact
6.83–6.97, random 6.59–6.91). All conditions recovered from a
damaged start (~13–18) to the same ~7 plateau; wrong-sector H
(7.73 at step 32) was modestly worse.

## F. Did the donor matter? — No measurable evidence that it did

Bypassing the whole graft assembly at step 32 scored 7.24–8.20,
the same range as the trained grafts. Connectors plus any
right-shaped sector — or none — explain the recovery. Restoration
is exact by construction and adds no information here.

## G. Recipient preservation

Trained grafts damage the recipient by ≈ +4.39…+4.47 at step 32
(score ~7.2 vs native 2.7731) — inside the frozen +4.50 ceiling
but, given no donor benefit, irrelevant to success: the graft is
both damaging and non-contributory under this protocol.

## H. Held-out

Not opened (`heldout_decision.md`). Historical sets untouched.

## I. Scientific verdict

Under this frozen protocol — Mamba blocks [11,13) into SmolLM at
[11,13), DenseSeam connectors, LR 3e-4, 96 steps — **the intact
foreign donor provides no donor-specific advantage.** The
engineering pipeline is fully validated (gates exact, donor
frozen, manifests complete); the scientific answer is negative.
This falsifies the strong hope that intact donor structure, by
itself, contributes beyond connector capacity — for this pairing,
location, and budget. It does not prove no pairing can work.

## J. Next action

A preregistered connector-routing diagnostic (does the connector
learn to bypass the graft?) before any larger pilot — see
`campaign_20_recommendation.md`. Requires fresh authorization.

## K. Publication

See `publication_verification.md` (final SHA, CI, PR #1 open).

## L. Ten-point summary

- The foreign graft was built exactly as frozen; all safety gates passed.
- The donor is genuinely structured at home (+0.22 over its scrambled self).
- After training, the intact donor *lost* to the scrambled donor in all 3 seeds.
- Random weights landed in the same range; bypassing the graft changed little.
- The connectors, not the donor's knowledge, explain the recovery.
- The recipient was still badly damaged (~7.2 vs 2.77) — with nothing to show for it.
- Because the success rule failed, the sealed held-out set stayed sealed.
- One runner bug was caught by the training contract before any scores existed; it is recorded.
- Engineering: complete success. Science: a clean, auditable negative.
- The transplant hypothesis is not disproved in general — it is unsupported here.
