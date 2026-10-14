# Campaign 9 Implementation Audit

## What changed vs Campaigns 7–8 (one variable)

Only the interface initialization. New method
`initialize_conditioned(..., "donor_native_geometry")`
(`src/equylaptatemps/alignment.py`, with `coloring_matrix`): orthogonal
variance-preserving seam pair → input seam whitened by the child's measured
input covariance, then **colored by the donor-native input covariance**
(SmolLM's own stream statistics at block 24, measured on TRAIN text with the
donor running natively) → output seam whitened by child-output covariance,
colored by recipient-native output covariance → scalar RMS gains under the
Campaign-8 Amendment-01 rule (skip/record if outside (0.05, 100)).

## Fairness of the matched control (Step 3.3)

The donor-native coloring matrices are identical for pretrained and twin
children (they are properties of the donor *architecture's* native stream,
measured from the frozen donor parent on the same 8 TRAIN windows); each
child's whitening uses its own measured stream — the equivalent procedure
with comparable information access. Trainable sets, budgets, data order,
optimizer, and evaluation are identical; the sector weights are the only
difference. The frozen Campaign 7 twin construction (native init, seed 1234)
is unchanged.

## Donor preservation (Step 3.2)

No donor parameter receives gradients (`assert_trainable_contract`; only
`input_seam`/`output_seam` trainable); the runner fingerprints both parents
before/after every run — all three Campaign 9 jobs report
`parents_unchanged: true`. The child executes the actual transplanted
SmolLM blocks [24,30) (`backbones.py` Llama path, RoPE in `StackSegment`);
no routing, ensembling, LoRA, distillation, or weight averaging anywhere.

## Tensor contracts (Phase 2 summary)

Sector input/output [B, T, 576]; host side [B, T, 768]; context limit 2048;
fp32; eager attention; additive causal mask. Full geometry analysis:
`representation_geometry.md`. Repository/Campaign-8 audit:
`repository_audit.md`.
