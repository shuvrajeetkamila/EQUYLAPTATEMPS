# Campaign 9 Representation Geometry (Phase 2)

## Donor tensor pathway (verified from code and config)

SmolLM-135M (`LlamaForCausalLM`, width 576, 30 blocks): embedding → blocks
0–29 (each: RMSNorm → GQA attention (9 Q / 3 KV heads, head-dim 64, RoPE
θ=10000) → residual → RMSNorm → SwiGLU MLP (1536) → residual) → final RMSNorm
→ tied head. Transplanted sector: contiguous blocks **[24,30)**, receiving the
native residual stream after block 23 and emitting the stream that natively
feeds the final norm. In the child (`model.py`), the sector instead receives
the input seam's output and feeds the output seam. Shapes: [B, T, 576] in/out;
RoPE positions derive from the attention mask inside `StackSegment`; eager
attention with an additive causal mask; fp32 in all experiments.

## Measured mismatch (development/TRAIN statistics, stored records)

| Quantity | Value | Source |
|---|---|---|
| Donor-native sector-input RMS | 196.50 | C7 `initial_statistics.json` |
| Child-fed sector-input RMS | 1.056 | same |
| Recipient-native suffix-input RMS | 1.434 | same |
| Child-fed suffix-input RMS | 0.922–0.952 | same |
| Trained graft relative update (pretrained / twin) | ≈19–58 / ≈0.19–0.47 | C7–C8 results |

Mismatch taxonomy: **scale** mismatch is real but largely cancelled by
per-token RMSNorm; **covariance/direction** mismatch is directly measured
(full covariance tensors are stored per stream) and has never been corrected
toward the *donor's* geometry — Campaign 8 whitened the child stream and
rescaled scalars, but the donor still received host-shaped geometry.
**Structural** mismatch (SSM state vs attention context) and **semantic**
mismatch remain hypotheses, not measurements.

## Campaign 9 intervention (feasibility: confirmed)

Construct each seam as: orthogonal pair → **whiten** the child-side stream by
its measured covariance → **color** with the target side's native covariance
(donor-native input covariance for the input seam; recipient-native output
covariance for the output seam), with the Amendment-01 gain rule (skip/record
out-of-range scalar gains). All statistics come from the same 8 unlabeled
TRAIN windows per child; the donor-native coloring matrices are identical for
pretrained and twin children (an architecture-level property), while each
child's whitening uses its own measured stream — an equivalent procedure with
comparable information access, isolating the sector weights as the only
difference. Implemented as `initialize_conditioned(..., "donor_native_geometry")`
with unit tests; no donor weights touched.
