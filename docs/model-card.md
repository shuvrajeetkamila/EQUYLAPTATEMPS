# Synthetic child model card

## Identity and intended use

**Name:** EQUYLAPTATEMPS synthetic surgery smoke child, version 0.1.0.

**Purpose:** validate heterogeneous architecture execution, full-block replacement, differentiability through frozen modules, frozen-weight integrity, and self-contained checkpoint loading. Not for answering real questions or deployment.

**Training/capability status:** tiny random parents; only new dense seams calibrated. No pretrained parent weights and no validated math/reasoning/language/code capability.

## Structure

Default seed 17, actual Hugging Face implementations:

| Component | Synthetic configuration |
|---|---|
| Recipient | Mamba, 4 blocks, width 16, 259 vocabulary IDs |
| Donor | Qwen2, 4 blocks, width 24, 311 vocabulary IDs |
| Recipient cut | `[1, 3)` — two complete blocks removed |
| Donor sector | `[1, 3)` — two complete blocks retained intact |
| Retained recipient | Native embedding, one prefix block, one suffix block, norm, tied output head |
| Interfaces | Two new dense projections with LayerNorm; 888 trainable parameters |
| Child | 20,376 unique parameters, 19,488 frozen |
| Position/state | Qwen2 native RoPE/causal attention plus Mamba native causal convolution/recurrence |
| Child tokenizer | Synthetic UTF-8 byte tokenizer; unrelated rotated byte contract recorded for donor |

Parent initialization seeds are 117 (recipient) and 217 (donor). The child sees recipient token coordinates only. It does not inherit a parent's learned semantics because neither parent was pretrained.

## Calibration and evaluation

40 steps of seam-only AdamW, learning rate 0.02, batch size 4, maximum 64 tokens, native blocks frozen in eval mode, gradient checkpointing enabled. Eight short arithmetic-format prompts have manually computed ground-truth answers; two disjoint prompts supply synthetic held-out diagnostics.

The checked-in [audit report](../reports/synthetic-smoke.json) records actual response-token NLL, all frozen/graft hashes, checkpoint weight hash, structure, full/bypass/zero ablations, and portable reload equality. It is a smoke test, not a benchmark or an instruction-tuned model.

In the recorded 40-step CPU run, training response-token NLL drops from about **5.640 to 4.367**. The native-graft bypass has almost the same held-out NLL as the full child. **That is not evidence of useful donor-skill transfer.** Random frozen blocks and seam fitting can move loss without teaching real arithmetic.

## Limitations and risks

- Random byte-level predictions can be nonsensical or replacement-decoded invalid UTF-8.
- Do not interpret arithmetic prompt formatting as demonstrated arithmetic correctness.
- Loss movement, tensor integrity, and an executable architecture do not establish specialist capabilities.
- The synthetic tokenizer is not a substitute for evaluating real parent tokenizer/representation mismatch.
- Checked-in evidence is CPU-only; GPU kernels, pretrained-size memory/performance, and long-context quality remain unvalidated.
- The child is a PyTorch module with its own CLI, not a Hugging Face `AutoModel`/hub release or a universally cache-compatible model.
- Local checksums are integrity checks, not signed attestations or proof of all historical training ancestry.

## Reproduction and distribution

```bash
equylaptatemps demo --out artifacts/demo --steps 40 --seed 17 \
  --report artifacts/demo-report.json
```

Use a new output directory. Reproducibility of floating-point values can vary with hardware/dependency versions; local save/load logits must remain identical under the same runtime. The report pins the observed environment.

Raw generated checkpoints are ignored by Git. One small, explicitly requested synthetic ZIP is included in `examples/` for portable verification; it contains no pretrained parent weights. Repository software is Apache-2.0. Real pretrained derivative checkpoints, if made later, are governed additionally by both parent licenses and their own training-data rights; this card does not grant permission to distribute them.
