# Campaign 3 — Preregistration (frozen before screening)

Frozen: 2026-10-13, before any Campaign 3 real-parent run. These criteria
must not be weakened after seeing results. Metric definitions are identical
to Campaign 2 (recipient-token token-NLL on development data; the historical
1.50× host language ceiling is retained because the audit found no error in
its calculation).

## Hypotheses

- **H1 (semantic-alignment hypothesis).** A causal, sequence-aware residual
  interface can supply the frozen Qwen sector with more useful contextual
  representations than the position-independent dense bridges of Campaign 2,
  improving development language retention and mathematics loss under matched
  data, seed, initialization, and training budget.
- **H0 (bypass/null hypothesis).** Any improvement comes from the interface
  learning to route around the donor sector, from extra trainable capacity,
  or from noise; the pretrained donor contributes nothing useful. H0 is
  tested with donor ablations and a matched random-sector trained twin.

## New diagnostic insight motivating the design

Campaign 2's stored position-binned statistics show the native Qwen sector
input RMS of 29.6 is dominated by the first position bin (87.8); steady-state
positions are ≈4.3, while the hybrid supplied a flat 0.82. The single scalar
gain of 36.4× fitted to the global RMS therefore plausibly overshoots
steady-state positions by ~8×. A position-independent scalar cannot express
"large at the start, moderate later"; a causal sequence mixer can. Matching
RMS still does **not** establish semantic alignment — only the gated
functional tests below can support an inheritance claim.

## Candidates (screen, seed 42, identical data/init/budget)

1. `c03_dense_cov_42` — Campaign 2 best bridge, retrained as control.
2. `c03_conv_cov_42` — dense + residual depthwise causal conv (kernel 5).
3. `c03_attn_cov_42` — dense + residual bottleneck causal attention (128, 4 heads).
4. `c03_gru_cov_42` — dense + residual bottleneck GRU (128).

All mixers are zero-initialized at their output: at initialization each
sequence candidate is numerically identical to the dense control. Parents
frozen; only `input_seam.*`/`output_seam.*` train; child's own next-token
loss on real TRAIN text and human GSM8K TRAIN answers only. No LoRA,
distillation, teacher targets, hidden-activation imitation, weight
averaging, ensembles, routers, or TEST selection.

## Frozen promotion gates

Primary metric: development macro token-NLL = mean(language, math) under the
host (Mamba) tokenizer, Campaign 3 development split (16 WikiText-2
validation windows + 16 GSM8K TRAIN-held-out rows, seed-42 selection).

A candidate is **screen-promoted** only if all hold:
- G1 language retention: dev language NLL ≤ 1.50 × host dev language NLL.
- G2 beats dense control: language NLL ≤ 0.95 × `c03_dense_cov_42` language
  NLL and math NLL ≤ 1.05 × its math NLL.
- G3a donor is used: full-child macro NLL ≥ 2% below both `bypass_graft`
  and `zero_graft` ablations of the same trained child.
- G4 mathematics signal: dev math NLL ≤ 0.90 × dense control math NLL.
- G5 integrity: frozen-parent fingerprints unchanged; bridge-only reference
  reconstruction reproduces identical logits in a fresh load.

**Final promotion** additionally requires, in the confirm phase:
- G3b beats a matched random-sector trained twin (same interfaces, data,
  seed, steps; donor blocks randomly initialized) on macro dev NLL.
- G6 seed replication: an independent seed (123) run satisfies G1.

Only a finally promoted candidate may receive held-out confirmation
(64-window WikiText-2 TEST bits-per-byte and GSM8K TEST mathematics, both
parents + matched random control, example-level records). If no candidate
is promoted, held-out candidate evaluation is NOT_RUN and the outcome is
classified honestly (expected C, or D if evidence is insufficient).
