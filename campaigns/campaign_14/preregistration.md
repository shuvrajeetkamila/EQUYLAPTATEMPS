# Campaign 14 — Preregistration (frozen before confirmatory tests)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-14
**Type:** reconstruction / damage-localization campaign. No donor
transfer comparison is run. No sealed held-out data is touched.

## Object and baseline

- Model: `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`,
  its own tokenizer, float32, CPU, 2 threads, eager attention.
- Evaluation (identical to Campaigns 7–13): GSM8K-train-derived
  dev 16 + WikiText-2 dev 16 (data seed 42), `evaluate_encoded`
  token NLL; macro = mean(language, math token NLL); lower better.
  Per-example scores recorded for every variant.
- Fixed diagnostic batch for output equivalence: the first 4
  encoded dev examples (validation order: encoded math then text),
  evaluated one at a time; max/mean |Δlogits| vs the native model
  and the batch's token-id SHA-256 are recorded.
- Environment/versions recorded in each job's records (torch
  2.5.1 CPU, transformers 4.48.3, python 3.11, ubuntu-22.04).

## Frozen tolerances (chosen now, before any Campaign 14 run)

- **Gate 1 — baseline reproducibility:** two native evaluations in
  the same job must agree to |Δmacro| ≤ 0.001 and logit
  self-difference exactly 0 on the diagnostic batch. (CPU float32
  deterministic evaluation; slack only for thread scheduling.)
- **Gate 2 — exact native reconstruction (machinery):** the
  native-placement child with seams bypassed must (a) contain
  every recipient tensor bit-identical, none missing, no extras
  besides bypassed (parameter-free) seams; (b) match native
  outputs: max|Δlogits| ≤ 5e-3, macro |Δ| ≤ 0.01. (Slack covers
  float32 accumulation-order differences between equivalent
  execution paths, nothing more.)
- **Gate 3 — function-preserving connector (the repair):** the
  same native-placement child with **zero-initialized residual
  seams** (`x + Linear(LayerNorm(x))`, projection weight and bias
  exactly zero → exact identity at initialization, trainable) must
  meet the same Gate 2 tolerances, with extras = the seam
  parameters (initial values recorded). Pass = reconstruction
  mechanism validated at native placement.
- Gates are applied **in order**; the trained stage (below) runs
  only if Gates 1–3 pass. Diagnostics (untrained variants) may all
  be measured in one job; conclusions obey the gate order.

## Variant matrix (diagnostic job; all untrained)

Recipient as both parents (`positive_control` recipes, labeled;
the graft is the recipient's own sector — a reconstruction test,
not transfer):

- **V0** native recipient (evaluated twice).
- **V1** cut [24,30) ← own sector [24,30), seams bypassed (Identity).
- **V2** cut [24,30) ← own sector, standard DenseSeams with the
  Campaign 13 initialization (`natgeom_in_trained_out`).
- **V3** cut [24,30) ← own sector, residual-identity seams.
- **V4** cut [11,13) ← own blocks [11,13) (the removed blocks
  themselves), residual-identity seams (minimal surgery at the
  Campaign 13 cut; native order/count preserved).
- **V5** cut [11,13) ← own sector [24,30), residual-identity seams
  (Campaign 13 placement with seams neutralized; combined
  relocation+removal+duplication+depth intervention, labeled as
  such).
- **V6** cut [11,13) ← own sector [24,30), standard DenseSeams,
  Campaign 13 initialization (= Campaign 13 condition F at
  initialization, untrained).

Per variant: per-example dev NLLs, macro, bypass/zero ablations,
graft activity, activation RMS summary, parameter-integrity audit
(identical/changed/missing/extra tensors vs the recipient,
tied-weight status), output-equivalence metrics, runtime.

## Trained stage (diagnostic; only if Gates 1–3 pass)

Seed 42, 32 steps, identical training configuration to Campaign 13:
- **V7** = V5 configuration + training (identity-preserving start).
- **V8** = V6 configuration + training (historical anchor;
  expected to reproduce Campaign 13's F ≈ 7.4577).
Report: initial damage (V5/V6 vs V0), post-training scores,
recovery (initial − trained). No claim of function preservation
is made from post-training scores.

## Decision rules (mechanical)

- `score_degradation = variant_NLL − native_NLL`;
  `relative_degradation_percent = 100 × degradation / native_NLL`.
- **Campaign outcome:** `RECONSTRUCTION_VALIDATED` if Gates 1–3
  pass; `DAMAGE_LOCALIZED_REPAIR_UNPROVEN` if Gates 1–2 pass but
  Gate 3 fails; `BLOCKED_AT_GATE_<n>` if Gate 1 or 2 fails (with
  downstream stopped). Damage attribution is reported from the
  variant deltas (seams: V2−V1; placement combined: V5−V3;
  total initial Campaign-13 configuration: V6−V0; training
  recovery: V6−V8 and V5−V7), each labeled with what it does and
  does not isolate.
- Promotion/held-out: not applicable; held-out remains
  `NOT_RUN_NO_PROMOTION`. Only the *mechanism* (identity-preserving
  reconstruction) can be "promoted" to recommended practice for
  future campaigns, via Gate 3.

## Integrity

Statuses PLANNED/RUNNING/COMPLETED/FAILED/INTERRUPTED/INVALID/
EXCLUDED_WITH_REASON in `run_registry.json`. Completed runs are
never rerun for a better score. All numbers are generated from
imported SHA-verified records by `scripts/publish_campaign14.py`;
verdicts computed mechanically. Amendments dated and pre-filed.
