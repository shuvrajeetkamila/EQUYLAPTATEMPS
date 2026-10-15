# Campaign 13 — Preregistration (frozen before confirmatory training)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-14
**Type:** same-family positive control (explicitly labeled; not a
cross-family claim). Implemented via the recorded
`Recipe.positive_control` opt-in (default off; Campaigns 1–12
protections unchanged).

## Hypothesis H13

In a genuinely compatible same-family setting, the surgery pipeline
(frozen donor sector + trained dense seams) preserves learned
structure: the intact donor sector (A) reproducibly outperforms
statistics-preserved scrambled versions of itself (B within-row,
C full-tensor) under identical treatment. If A shows no
reproducible arrangement sensitivity even here, the pipeline/regime
— not only the cross-family pairing — is implicated in the
Campaign 11–12 nulls.

## Frozen pair and setup

- Recipient: `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`
- Donor: `HuggingFaceTB/SmolLM-135M-Instruct@fcc320f490e08fdb4b99d935b2c58d40bf35b0d0`
  (official instruction-tuned derivative of the same base; probe
  job `38766385142`: configs identical in all architectural fields,
  sector weight delta 5.64% mean relative RMS, parents unchanged)
- Recipient cut [11, 13); donor sector [24, 30); donor frozen;
  interface `natgeom_in_trained_out`; macro development NLL
  (lower better); child uses the recipient tokenizer throughout.
- Host reference = recipient's native dev scores as recorded by
  each job (`native_development.json`): probe values language
  4.0773, macro 2.7731.
- Data, optimizer, training configuration: exactly the Campaign
  12 registry configuration (AdamW lr 3e-4, batch 1, max len 512).

## Conditions (all fresh runs)

- **A** intact Instruct sector. **B** within-row shuffle (seed
  20241014). **C** full-tensor shuffle (seed 20241014, Campaign
  11–12 implementation unchanged). **D** random twin
  (`random_sector_seed` 1234), reference. **E** bypass/zero
  ablations inside every run (diagnostic). **F** mechanical
  self-reconstruction (donor := recipient checkpoint, one seed-42
  run, separate registry/job; integrity check only, never evidence
  of transfer).
- Runs: 32 steps × seeds {42, 123, 31415} × {A,B,C,D} = 12
  (`c13_{pre,row,full,rand}_{seed}`); 96 steps × seeds {42, 123}
  × 4 = 8 (suffix `_96`), run **only if all 32-step B/C runs pass
  validity** (else the campaign stops at Outcome D review).
  96-step panel is diagnostic; classification rests on 32 steps.

## Validity (frozen; per B/C run, from its records — wording exact)

(1) every eligible tensor reports exact multiset preservation
(B additionally: every row's multiset preserved); (2) every
eligible tensor's fingerprint changed; (3) trained graft relative
update ≥ 0.05; (4) bypass and zero ablation macros finite and
above intact; (5) `parent_weights_unchanged` true. Publisher code
implements exactly this list (unit-tested against hand-computed
examples before training, per the Campaign 12 lesson); the
stricter bypass<zero ordering is reported per run as a
non-gating diagnostic. D's activity is reported; beating an
inactive D is not evidence. If any B/C run is invalid → Outcome D
unless a pre-inspection amendment defines a repair.

## Decision framework (frozen; canonical protocol has no
positive-control rule, so this is preregistered per brief Part 5/8)

Benefit vs X = 100·(L_X − L_A)/L_X at matched steps+seed.
**Learned-structure gate** (canonical thresholds): A beats each
of B and C by ≥2% at seed 42 and ≥1% at seeds 123 and 31415
(32 steps). **Causal support:** A's intact macro below its bypass
macro at all three primary seeds. **Usefulness gate G-C**
(canonical values, host = native recipient): A's language ratio
≤ 1.50× host language NLL and macro ≤ 6.00 at all primary seeds.

- **Positive control PASSES** ⟺ gate + causal support + validity.
- **Outcome A:** positive control passes and G-C passes.
- **Outcome B:** positive control passes; G-C fails. (Pipeline
  detects/preserves learned structure in a compatible setting;
  child not useful. Cross-family transfer remains unproven.)
- **Outcome C:** validity holds; positive control fails.
- **Outcome D:** a B/C control invalid, or other documented
  validity failure — inconclusive.

**Arrangement-sensitivity diagnostic** (descriptive, frozen): if
the gate fails, report `arrangement_sensitivity` as
`adverse_detected` when benefits vs B and vs C are negative at all
three primary seeds with pooled mean ≤ −2.0% (pipeline registers
arrangement, direction harmful for this endpoint), else `none`.
This never changes the outcome letter or promotion.

**Promotion / held-out:** the sealed cross-family TEST is not
opened by this campaign under any outcome:
`NOT_RUN_NO_PROMOTION` is recorded (a same-family child is not the
promotion candidate the sealed protocol governs).

## Statistics and reporting

All seed-level benefits reported; descriptive mean/median/range
over the three primary seeds (n=3, no significance claims).
Every run reported including FAILED/BLOCKED; no selective reruns;
interrupted jobs resume via re-trigger (runner skips
verified-complete runs). Verdict computed mechanically by
`scripts/publish_campaign13.py` from imported, SHA-verified
records. Amendments dated and pre-filed; TEST data never used.
