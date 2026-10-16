# Campaign 11 Repository & Evidence Audit (Phase 0, 2026-10-14)

## Live state

- Branch `arena/97efdf4a-equylaptatemps` (session-fixed); HEAD restored
  non-destructively to the verified remote head
  `42d628458afc36eb22cfe69e52e0a3dd39268e65` after the recurring
  session-restore artifact (tree verified clean; nothing lost).
- PR #1 OPEN, not merged: "Campaigns 1-10 complete — awake-control
  experiment, fragile donor-specific effect, Outcome B"; CI success.
- Resources: no GPU, ~3.8 GiB RAM locally; real runs on GitHub Actions CPU
  runners (proven pipeline, Campaigns 2–10).

## Campaign 10 verification (primary records)

All ten Campaign 10 macros recomputed from raw `results.json` match the
published `results.json`; comparisons confirmed: 32 steps B vs D
+10.43%/+4.42% (seeds 42/123), B vs C +5.03%/+0.97%; 96 steps B vs D
+1.17%, B vs C −1.69%. The exploratory awake run (total scale 591.27,
pre-amendment) is preserved and labeled `D_exploratory`; the confirmatory
amended run (586.44, calibrated=true, `c10_out_awake_42_amd1`) is the one
used in comparisons. Outcome B; held-out `NOT_RUN_NO_PROMOTION`.

## Donor sector tensor inventory (Step 0.3, SmolLM blocks [24,30))

Per Llama block (no biases in this checkpoint): **eligible 2D matrices** —
`self_attn.q_proj` [576,576], `k_proj` [192,576], `v_proj` [192,576],
`o_proj` [576,576], `mlp.gate_proj` [1536,576], `up_proj` [1536,576],
`down_proj` [576,1536] (7 per block, 42 total). **Excluded 1D** —
`input_layernorm.weight`, `post_attention_layernorm.weight` [576] (2 per
block, 12 total): per-channel norm gains whose permutation would silently
re-wire normalization semantics rather than destroy matrix organization;
they are preserved and the exclusion is recorded in every control record.
A within-tensor permutation preserves each eligible tensor's exact value
multiset (hence mean/variance/RMS/min/max/quantiles and Frobenius norm)
while destroying learned arrangement (row/column structure, feature
correspondences). Architecture, shapes, parameter counts, and execution
path are unchanged; RoPE lives in the segment, untouched.
