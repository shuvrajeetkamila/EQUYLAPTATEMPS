# Campaign 4 — Preregistration (frozen before training)

Frozen 2026-10-13, before any Campaign 4 real-parent run.

## Question and hypotheses

**Question.** When the architecture makes the donor pathway structurally
necessary, can intact pretrained Qwen blocks [20,22) contribute useful,
measurable capability to the Mamba-host child?

- **H1.** A mandatory-use child will beat its matched random-sector twin and
  degrade when the donor is removed/neutralized.
- **H0.** Even under structural forcing, bridges compensate, the donor stays
  a passenger or harms performance, and random sectors match or beat it.

## Candidates (all: host pathway/tokenizer as Campaign 3; parents frozen;
only interface parameters train; child's own TRAIN next-token loss only)

1. `c04_delta_attn_42` — Campaign 3 attention seams + **donor-delta pathway**:
   the suffix receives only (donor output − donor input). An identity donor
   yields a zero signal, so success requires real donor transformation.
2. `c04_bn128_attn_42` — **128-wide bottleneck seams** (norm → down → causal
   attention mixer → up; 893,696 trainable) capping seam-to-seam capacity.
3. `c04_bn128_delta_attn_42` — bottleneck + donor-delta combined.

Each has a matched random-sector twin (identical architecture, parameters,
data, seed, steps; sector seed 2024) in the twins phase. Covariance
initialization as Campaigns 2–3 (bottleneck variant whitens `down`, gains
on `up`); no teacher targets of any kind.

## Protocol

Identical to Campaign 3 for comparability: seed-42 data (32 GSM8K TRAIN +
32 WikiText-2 train windows for training; 16+16 development), 32 AdamW
steps, lr 3e-4, max length 512, CPU float32. Primary metric: development
macro token-NLL = mean(language, math) under the host tokenizer.

## Frozen promotion gates

- **Gate A (integrity):** one executable path; pinned intact donor;
  fingerprints unchanged; bridge-only reconstruction with identical logits;
  causality tests pass.
- **Gate B (pretrained benefit):** pretrained macro ≤ 0.98 × matched random
  twin macro (≥2% relative). Campaign 3's pretrained-vs-random gaps were
  ~12–23% in the wrong direction, so 2% is well above noise resolution.
- **Gate C (donor dependence):** full-child macro ≤ 0.98 × both
  `bypass_graft` (donor-removed) and `zero_graft` (donor-neutralized) macro.
- **Gate D (useful performance):** language NLL ≤ 1.50 × host language NLL
  (historical ceiling, same metric) AND macro ≤ 5.7396 (Campaign 3 dense
  control on the identical development protocol) — dependence achieved by
  destroying performance does not count.
- **Gate E (replication):** for any candidate passing A–D, an independent
  seed pair (seed 123 pretrained + twin) must preserve Gate B's direction.

Only a candidate passing A–E may receive held-out confirmation. Otherwise
held-out status is `NOT_RUN_NO_PROMOTION`. Thresholds will not be relaxed
after results are seen; inconclusive (D) is reserved for evidence failures,
not negative results.
