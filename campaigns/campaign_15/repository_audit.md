# Campaign 15 — Repository Audit (Stage 1)

**Date:** 2026-10-15
**Repository:** `shuvrajeetkamila/EQUYLAPTATEMPS`, branch
`arena/97efdf4a-equylaptatemps`
**Verified starting state:** local HEAD = remote HEAD =
`6ba746deeecdd72ce8835c07e5cec9b0a3054d84` (Campaign 14 complete);
working tree clean after recovery from the known session-restore
artifact (non-destructive reset to remote; no history rewritten).
PR #1 OPEN, unmerged, retitled for Campaigns 1–14. Latest CI
`38799869698` success; 284 tests passed locally at Campaign 14's
close (re-verified below where relevant).

## What is confirmed by executable code

- **Old connector** (`src/equylaptatemps/model.py`, `DenseSeam`):
  `Linear(LayerNorm(x))`, Xavier-initialized; Campaign 13/15
  condition B additionally applies `natgeom_in_trained_out`
  (`initialize_conditioned`): input seam covariance-colored,
  output seam generic Xavier. Verified at runtime: a fresh
  `DenseSeam` is **not** the identity map.
- **New connector** (`ResidualDenseSeam`, added in Campaign 14,
  commit `3944c79`): `x + Linear(LayerNorm(x))` with projection
  weight and bias exactly zero. Verified at runtime this audit:
  output equals input **bitwise** (`torch.equal`) at
  initialization; the class rejects width-changing shapes.
  Installed via `install_interfaces` architecture
  `"residual_identity"`; seams remain the only trainable
  parameters (assert_trainable_contract passes).
- **Both connectors expose exactly 667,008 trainable parameters**
  (two 576-wide seams) — the comparison is parameter-fair.
- **Native reconstruction path:** `child_from_parents` +
  `StackSegment` (positions derived from the attention mask =
  global positions; no per-segment restart). With seams bypassed
  or residual, the child computes the native function (Campaign 14
  evidence below).
- **Changed-placement path:** cut [11,13), sector [24,30) →
  prefix 11 + graft 6 + suffix 17 = 34 blocks; blocks 11–12
  removed; blocks 24–29 duplicated. Confirmed in Campaign 14
  parameter audits (309/309 tensors identical to their source
  blocks; removed/duplicated lists exactly as designed) — no
  tensor-mapping defect exists.
- **Training loop** (`campaign_training.fit_resumable`): AdamW
  (lr 3e-4, wd 0), gradient clipping 1.0, batch size 1, data order
  from a dedicated `torch.Generator(seed=experiment seed)` —
  **data order is exactly paired across connector conditions at
  the same seed**. Checkpoints (parameters + AdamW state + RNG)
  every 8 steps; resume verifies config/data/code/frozen hashes;
  frozen-weight fingerprint is asserted before and after fitting.
  Only intermediate *training* diagnostics exist; scheduled
  **validation at intermediate steps requires the Campaign 15
  runner's segment-driving** (see preregistration; no library
  change).
- **Evaluation:** `evaluate_encoded` token NLL; macro = mean of
  language/math group token NLLs (lower better); GSM8K-train dev
  16 + WikiText-2 dev 16, data seed 42 — identical across
  Campaigns 13–15. Held-out/test never touched.

## What is supported by saved raw evidence

Independently recomputed from Campaign 14's SHA-verified records
(`diagnose_38792029752`, bundle `273470b8…`; `trained_38796151693`,
bundle `7d5f5b2e…`):

- Native recipient macro **2.7731**, repeated evaluation diff 0.0.
- V1 (native placement, seams bypassed) and V3 (residual seams):
  macro 2.7731, maximum logit difference **0.0**, 273/273 tensors
  identical — exact reconstruction.
- V2 (old seams at native placement): **24.4656** (+21.69).
- V5 (Campaign 13 placement, residual seams): **12.0150** (+9.24);
  V6 (historical configuration, untrained): **14.2481**.
- Trained seed-42 pair: new connector 12.0150 → **8.4623**; old
  connector 14.2481 → **7.4578**, the latter replicating Campaign
  13's independently produced F record (7.4577) to 1e-4.

## What is merely asserted / remains unknown

- The old connector's trained advantage at seed 42 (7.46 vs 8.46)
  is **one paired observation**; seed generality is unknown — the
  question Campaign 15 exists to answer.
- Old-connector initial scores at seeds 123/31415 were never
  measured (Campaign 13 trained from unrecorded initial states);
  Campaign 15 records them for every seed.
- Whether the segmented-driving evaluation (interrupt at
  8/16/32/64, evaluate, resume) reproduces uninterrupted training
  bit-for-bit is an instrumentation assumption Campaign 15's
  smoke test checks empirically (loss-trace continuity and the
  step-96 double evaluation).

## Can Campaign 15 run without changing the historical experiment?

Yes. Condition B is byte-identical to Campaign 13/14's historical
configuration (same runner lineage, `natgeom_in_trained_out`,
667,008 trainable seam parameters); condition A differs **only**
in the connector class/install architecture. Placement, data,
optimizer, budget, scoring, and checkpoint policy are unchanged.
Required additions are confined to a derived runner
(segment-driving + per-checkpoint evaluation), registries,
workflows, publisher, and tests.
