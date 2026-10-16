# Campaign 17 — Native-Depth Specification

**Architecture:** SmolLM-135M (`LlamaForCausalLM`, config-verified):
embedding 576-wide (tied to the LM head), 30 llama decoder
blocks, final RMSNorm. A **block** = one `LlamaDecoderLayer`
(input norm, GQA attention, post-attention norm, MLP). Blocks
share no parameters with each other; the only tied pair is
embedding ↔ lm_head, which lives outside the block sequence.

**Native depth** = 30 executed blocks in the recipient's original
order 0…29, each exactly once.

A placement is **native-depth** iff:
1. Executed block count == 30 (detected by segment lengths).
2. The executed sequence is a permutation of the original block
   identities `block:0 … block:29` (detected by per-block tensor
   fingerprints matched against the source checkpoint's blocks).
3. Every deviation from identity order is declared in the
   placement manifest (moved blocks listed; undeclared deviation
   = structural-integrity failure).
4. Embedding, final norm, and tied head are the recipient's own
   modules (identity-checked); tied relationship preserved.
5. Boundary shapes (residual width 576 at every block input/
   output) verified per block; any mismatch aborts assembly.
6. Trainable parameters, if any, are confined to the declared
   connector modules; all block parameters are the source
   tensors (object-identical for untouched positions, value-
   identical for replaced positions).

**Manifest (machine-readable, per placement):** for each executed
position: stable identity (`block:<original index>`), original
index, final index, source role (recipient/graft-donor), tensor
fingerprint, parameter count, action
(untouched/replaced/moved/exchanged); plus depth before/after,
duplicated list (must be empty), removed list (must be empty).
