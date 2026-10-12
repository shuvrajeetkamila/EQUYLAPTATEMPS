# Architecture and surgical contract

## A literal whole-block splice

Let recipient `R` have width `d_R` and `n` blocks, donor `D` have width `d_D` and `m` blocks. Choose half-open intervals `[r0, r1)` and `[d0, d1)`.

```text
h0 = E_R(recipient_ids)
h1 = R[0:r0](h0)
h2 = P_RD(LayerNorm_R(h1))
h3 = D[d0:d1](h2)
h4 = P_DR(LayerNorm_D(h3))
h5 = R[r1:n](h4)
logits = Head_R(Norm_R(h5))
```

`D[d0:d1]` contains complete native decoder/state-space blocks, not selected coordinates of their weight matrices. The recipient blocks `[r0:r1]` are absent from the child. Native recipient embedding, retained prefix/suffix, norm, and vocabulary head are frozen. Unused donor blocks, embedding, final norm, and head are absent.

Block objects are transplanted directly from the loaded parent modules. The source-sector tensor hash is checked against the child's sector, and all frozen state tensors are hashed before and after seam calibration. The comparison is at the selected common loading precision: a dtype conversion at parent load time is not weight interpolation.

The new child has a heterogeneous computation graph. This is architectural composition/transplantation, **not numerical weight-space model merging**. The implementation does not claim architectural surgery is scientifically established to preserve specialist abilities.

## New interface parameters

Each seam is a new LayerNorm followed by a full-sized dense affine projection:

```text
P_RD: d_R -> d_D
P_DR: d_D -> d_R
```

With both projection biases and both normalizers, the trainable parameter count is:

```text
2 * d_R * d_D + 3 * (d_R + d_D)
```

The tiny 16/24-width demo therefore trains 888 interface parameters. “Full-rank” describes the unfactorized parameterization, not a guarantee that a trained matrix has maximal algebraic rank. There are no LoRA matrices, no updates attached to parent linear layers, and no bottleneck factorization.

There is **no residual bypass around the donor**, no inference gate that can skip it, no second model predicting its own answer, no combination of parent logits, and no arithmetic over corresponding parent weights. Explicit `bypass_graft` and `zero_graft` modes exist only for evaluation diagnostics and are rejected during seam training.

## Objective

Only new seam parameters are optimized:

```text
minimize mean ground-truth response-token cross entropy
subject to every retained recipient/donor tensor remaining unchanged
```

Gradients traverse the frozen donor and recipient suffix. Freezing their parameters does not detach their computations. The recipient prefix normally needs no gradient graph because all its inputs and parameters are frozen.

Targets come from the supplied JSONL answers, not either parent's outputs. There is no teacher inference, feature-matching objective, or teacher-logit term. Supervised fitting of new boundary matrices is **not** distillation.

Native frozen modules remain in evaluation mode even while seams train, preventing parent dropout from changing the execution distribution. Gradient checkpointing recomputes native blocks without detaching seam gradients. Re-running `train` uses a fresh optimizer; optimizer-state resume is not implemented.

## Different tokenizers

Only recipient IDs are embedded and emitted. The donor sector never receives token IDs. It receives projected recipient hidden vectors and the same causal time coordinates.

This solves the *tensor interface* problem without pretending that a token ID from one vocabulary has meaning in another. It does **not** solve the *semantic interface* problem automatically: donor blocks were trained on activations produced by their own tokenizer, embedding, and earlier blocks. Those are absent in the child. Projection calibration may not recover that distribution, especially when token boundaries differ substantially.

The parent tokenizer contracts are fingerprinted independently and must differ for the strict local-parent workflow. The donor tokenizer is used for provenance and optional native-parent evaluation, not in the child forward pass. The recipient tokenizer is bundled with the child and verified on reload. Existing EOS may serve as padding; vocabulary resizing is forbidden.

## Native runtime details

- **Qwen2:** complete eager-attention decoder blocks, native grouped-query attention, native RMSNorm/MLP, config-derived RoPE, a four-dimensional additive causal/key-padding mask, no KV cache. RoPE frequencies are regenerated in fp32 after dtype/device moves because they are nonpersistent, config-derived buffers. Sliding-window attention is not supported.
- **Mamba:** complete native Mamba blocks with their selective state-space recurrence and causal convolution, explicit attention masks, no SSM cache. The stock sequential fallback works on CPU without optional Mamba kernels. Its fp32 residual behavior is handled at seam/head dtype boundaries.
- Right-padded, nonempty batches only; no left padding or mixed cache state.
- The child uses the minimum supported Qwen2 context limit present in its two parent configs. Position-space compatibility does not imply inherited long-context quality.
- Any transplanted transformer attention retains quadratic full-context attention work. This child is not a pure linear-complexity Mamba model.
- Both parents must use one device and a common floating dtype. Quantization, multi-device sharding, arbitrary remote model code, and other architectures are not supported.
- Portable reload instantiates only retained native blocks, not full parents, then uses strict safetensors loading. Tied recipient embedding/head weights are preserved.

## What “complete strangers” can and cannot mean

The allowlist records Mamba and Qwen weight-lineage roots and public project references. Recipes reject the same family, shared declared roots, unreviewed model IDs, placeholder revisions, and mixed synthetic/pretrained identities. Local build checks actual config architectures, sector bounds, and tokenizer fingerprints. Acquisition records pin exact commits and check available Hub LFS hashes; later builds validate their local inventories.

These checks cannot infer an undisclosed checkpoint's historical training ancestry from its tensors. Local download records can be edited and are not signed attestations. Different architectures do not imply different training corpora or zero shared design ideas. Manually supplied revision declarations are not remotely verified. Review source model cards and actual artifacts rather than treating a successful validation command as a mathematical proof of provenance.

## Why “math sector” is not a fact

Contiguous layer ranges contain distributed, context-dependent features. Selecting decoder layers from a math-oriented model does not prove those layers alone implement math. Frozen embeddings/head and omitted donor prefix/suffix may be essential.

`search` provides a practical test: calibrate independent seams around several complete sectors using training data, rank them on disjoint validation data, then evaluate the selected child on an untouched test set against both parents. A poor graft, a non-helpful bypass ablation, or lost recipient skills is a valid negative result. Do not rename an unvalidated splice as a successful capability fusion.

The initial supported backends are pinned to Transformers 4.48.3. Upgrading library internals requires repeating native-parent equivalence, causality, gradient, precision, and serialization tests; this is a research dependency pin, not a production security certification.
