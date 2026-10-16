# Campaign 17 — Repository Audit (Stage 1)

**Date:** 2026-10-16. **Starting state verified:** local HEAD =
remote HEAD = `b6006108f86c05ba9d810e37b9de96a24ea98aa1`
(Campaign 16 final; recovered from the known session-restore
artifact non-destructively), tree clean, PR #1 OPEN unmerged.

## Verified history (recomputed from SHA-verified records)

- Campaign 13 self-control: native 2.7731 vs child 7.4577.
- Campaign 14: exact native reconstruction (bitwise); connector
  damage +21.69; placement damage +9.24; repair exact.
- Campaign 15: TRAINING_DEPENDENT (old wins @32 3/3, new wins @96 3/3).
- Campaign 16: LEARNING_RATE_SENSITIVE — step-32 paired means
  −2.59 / −0.99 / +0.17 across LR {9e-5, 3e-4, 9e-4}; baseline
  cells reproduce Campaign 15 within 7.2e-05.

## Actual recipient architecture (from code and config)

SmolLM-135M = LlamaForCausalLM: embedding (tied to lm_head),
**30 decoder blocks** (each: input_layernorm, self-attention with
GQA, post-attention layernorm, MLP; RoPE positions derived from
the attention mask, i.e., global token positions), final RMSNorm,
tied LM head. No per-block state buffers beyond parameters;
depth = number of decoder blocks = 30.

## Exact historical placement algorithm (the defect)

`child_from_parents` always builds prefix = recipient blocks
[0, cut.start), graft = donor blocks [sector], suffix = recipient
blocks [cut.stop, 30). With the Campaign 13 registration
(cut [11,13), sector [24,30)) this *necessarily*: deletes blocks
11–12, duplicates donor blocks 24–29 (they also remain inside the
suffix), reorders execution (24–29 run before 13–23), and grows
depth 30 → 34. Deletion/duplication/reordering are artifacts of
registering a **non-depth-matched** sector replacement in a
three-segment assembler — not essential to the surgery concept.

## Smallest defensible design change

Constrain placement to **depth-preserving execution orders**: the
child executes a *permutation* of the recipient's 30 blocks
(identity for reconstruction/replacement; a declared swap for
the exchange diagnostic), assembled by a new engine
(`native_placement.py`) that fails loudly on any non-permutation
order or shape mismatch, emits a per-block structural manifest,
and reuses the established segments/seams/evaluation unchanged.
The historical path remains available untouched for comparison.
