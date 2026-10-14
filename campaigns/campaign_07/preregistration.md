# Campaign 7 Preregistration — Cross-Pairing Replication (FROZEN)

**Project:** HETEROGENEOUS AI MODEL SURGERY  
**Frozen:** 2026-10-14, before any Campaign 7 training outcome was observed.  
**Question:** Can an intact pretrained sector from a different architecture family
contribute useful, measurable capability in a **new host–donor pairing**
(Mamba-130M host × SmolLM-135M/Llama donor), under controlled comparison against
a matched random-sector twin?

## Frozen pairing (no silent substitution)

| Role | Checkpoint | Pinned revision | Family |
|---|---|---|---|
| Host (recipient) | `state-spaces/mamba-130m-hf` | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | Mamba SSM, width 768, 24 blocks |
| Donor | `HuggingFaceTB/SmolLM-135M` | `1d461723eec654e65efdc40cf49301c89c0c92f4` | Llama (GQA 9/3 heads, head-dim 64, SwiGLU f=1536, RoPE θ=10000, RMSNorm), width 576, 30 blocks, tied embeddings, vocab 49152, Apache-2.0 |

Selection rationale and rejected candidates: `pairing_selection.md`. Any
checkpoint change requires a dated amendment **before** further training.

## Frozen child design

- Topology: Mamba embedding + blocks [0,11) → **input seam** → intact contiguous
  SmolLM blocks **[24,30)** → **output seam** → Mamba blocks [13,24) → Mamba final
  norm → tied Mamba head. Recipient cut [11,13). No router, ensemble, residual
  bypass, weight averaging, distillation, or LoRA anywhere in the primary design.
- Interfaces: the only trainable tensors are two full-size dense seams
  (LayerNorm + Linear 768→576 and 576→768, ≈0.89M parameters), `DenseSeam`
  defaults, initialization kind `random`, seeded by the recipe seed. Parent
  weights frozen; integrity fingerprints verified before/after every run.
- Primary comparison: pretrained sector vs **matched random twin** — identical
  shape/native Llama blocks, architecture-native initialization
  (`initialization_seed=1234`), identical seams, data, budget, and seed 42.
- Protocol identical to Campaigns 3–6: 32 optimizer steps, batch 1, lr 3e-4,
  max_length 512, gradient checkpointing; train = GSM8K train 32 + WikiText-2
  train 32 windows; dev = GSM8K dev 16 + WikiText-2 validation 16 windows;
  **macro = mean(language, math) dev token NLL** under the host tokenizer.

## Frozen gates

- **G-B (primary, pretrained advantage):** `macro_pretrained ≤ 0.98 × macro_twin`.
- **G-C (absolute usefulness):** `language_ratio_vs_host ≤ 1.50` (ceiling unchanged
  since Campaign 2) **and** `macro_pretrained ≤ 6.00`.
- **Donor-use diagnostics (not gates):** bypass/zero-graft macros and graft
  relative update are reported; dependence alone is never claimed as benefit.
- **Promotion to held-out:** G-B ∧ G-C at seed 42 **and** seed-123 replication
  with benefit ≥ 1% (`macro_pre ≤ 0.99 × macro_twin`). Only then may held-out
  (WikiText-2 test BPB + GSM8K test, frozen Campaign-2 protocol) be opened,
  exactly once. Otherwise held-out remains `NOT_RUN_NO_PROMOTION`.

## Frozen outcome mapping (exactly one)

- **Outcome A:** promotion gates pass **and** held-out confirms usefulness
  (held-out language BPB ratio ≤ 1.50 vs host, computed once).
- **Outcome B:** G-B passes with replication (donor-specific contribution beyond
  the matched control) but G-C or held-out usefulness fails — child not useful.
- **Outcome C:** valid experiment; G-B fails (no reliable pretrained advantage).
- **Outcome D:** inconclusive — a documented technical/data/compute/measurement
  limitation prevents a valid comparison (e.g., pinned-checkpoint acquisition or
  integrity failure). Must not be used to relabel a clear negative.

## Stopping rules and amendments

Screen = 2 runs (pretrained + twin, seed 42). Replication = 2 runs (seed 123),
registered by amendment **after** screen analysis and **before** running it,
regardless of screen direction. No sector, width, budget, or threshold changes
after seeing outcomes; any change is a dated amendment filed before the affected
run. Failed runs are preserved, never erased or silently retried under a new ID.

## Integrity

Parent fingerprints before/after; donor sector tensor hashes recorded; random
twin construction seed fixed above; TEST data never used for selection.
