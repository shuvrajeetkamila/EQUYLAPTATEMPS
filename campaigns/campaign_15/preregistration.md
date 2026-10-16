# Campaign 15 — Preregistration (frozen before any Campaign 15 run)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-15
**Type:** controlled validation of the repaired connector under
training (same-family self-control). No heterogeneous transfer,
no held-out data.

## Primary question

At the Campaign 13 placement (recipient cut [11,13), own sector
[24,30), 34 blocks), does the residual-identity connector produce
a more stable or better trained self-control than the historical
connector across seeds {42, 123, 31415}?

## Conditions (only the connector differs)

- **A (new):** `residual_identity` interfaces (Campaign 14
  `ResidualDenseSeam`, exact identity at init; 667,008 trainables).
- **B (old):** `layernorm_linear` DenseSeams with
  `natgeom_in_trained_out` initialization (the exact Campaign 13
  configuration; 667,008 trainables).
Identical: recipient checkpoint (SmolLM-135M @ `1d461723…`) in
both parent roles (`positive_control`), tokenizer, placement,
training/dev data (data seed 42; GSM8K train 32/dev 16, WikiText-2
32/16, 192 chars), optimizer (AdamW lr 3e-4, wd 0, clip 1.0,
batch 1, max_length 512, gradient checkpointing), data order
(dedicated generator seeded per experiment — exactly paired
across conditions), precision float32, 2 threads, evaluation
(`evaluate_encoded`, macro of language/math token NLL; lower
better), final-step checkpoint policy.

## Runs and checkpoints

Six primary runs: {A,B} × seeds {42,123,31415}, each trained to
96 steps, validation evaluated at steps 0, 8, 16, **32 (primary
endpoint)**, 64, 96. Training is driven in segments
(`interrupt_after` at 8/16/32/64, evaluate, resume) using the
library's hash-verified resume; the smoke test validates this
instrumentation before the panel. Step-0 scores are recorded per
run (initial_validation) together with graft activity.

## Gates (frozen; Campaign 14 tolerances reused)

- **Gate A (regression):** a fresh Campaign 14 diagnostics run on
  Campaign 15's code state must reproduce: native repeat |Δmacro|
  ≤ 0.001; V1/V3 macro |Δ| ≤ 0.01 and max|Δlogits| ≤ 5e-3 vs
  native, all audited tensors identical. Failure ⇒ stop the
  panel (IMPLEMENTATION_FAILURE path).
- **Gate B (initialization):** at the changed placement, for each
  seed, step-0 delta0 = old − new > 0, and mean delta0 ≥ 1.0.
- **Gate C (primary):** at step 32, delta32 = old − new > 0 for
  all three seeds.
- **Gate D (duration):** compare delta32 with deltas at 64/96
  (mechanical rule in `decision_rules.md`).
- **Gate E (readiness):** recommendation mapping only (decision
  memo / `campaign_16_readiness.md`); not a pass/fail gate.

## Integrity rules

Run statuses PLANNED/RUNNING/COMPLETED/FAILED/INTERRUPTED/INVALID/
EXCLUDED_WITH_REASON; failed attempts preserved in
`failure_registry.json` with classification; completed runs are
never rerun for a better score; resumes only via the library's
hash-verified mechanism; parents verified unchanged per run;
held-out remains `NOT_RUN_NO_PROMOTION`. Outcome categories and
their mechanical precedence: `decision_rules.md`.
