# Campaign 13 — Model Pair Selection (Positive Control)

**Date:** 2026-10-14
**Status:** pair selected on documented criteria; exact donor
revision to be pinned by the exploratory probe (below) and frozen
in the preregistration before any confirmatory run.

## Selected pair

- **Recipient (host):** `HuggingFaceTB/SmolLM-135M` @
  `1d461723eec654e65efdc40cf49301c89c0c92f4` (the Campaign 7–12
  donor, now cast as recipient; exact revision already recorded).
- **Donor:** `HuggingFaceTB/SmolLM-135M-Instruct` @ *(revision
  resolved by probe job, then frozen)* — HuggingFaceTB's official
  instruction-tuned derivative of the same SmolLM-135M base.
- **Architecture (both):** Llama, hidden 576, 30 layers, 9 heads /
  3 KV heads, intermediate 1536, vocab 49152, tied embeddings,
  RMSNorm, RoPE — identical config expected; the probe verifies
  `config.json` equality field-by-field and records any difference.
- **Tokenizer:** expected identical files (the Instruct release adds
  a chat template, not new vocabulary); the probe compares
  tokenizer fingerprints and file hashes and reports the result.
  The child always uses the recipient tokenizer.
- **Graft:** donor sector [24, 30) into recipient cut [11, 13) —
  the same spans as Campaigns 7–12, now same-width (576→576 seams).
  Sector = 6 blocks ≈ 21.24M parameters (as measured in Campaign
  12 validation). Child = 34 Llama blocks + trained seams.

## Why learned-structure transfer is reasonably expected here

The two checkpoints share architecture, tokenizer, initialization
lineage, and almost all training; the donor sector differs from the
recipient's own corresponding weights by (i) the base model's
learned organization (identical lineage) and (ii) an
instruction-tuning delta applied to every layer. If the surgery
pipeline (frozen sector + trained dense seams, 32–96 steps) can
preserve *any* learned arrangement, this is the friendliest
possible setting: no width change, no family change, no tokenizer
change, neuron alignment inherited from a common ancestor. The
scramble controls (within-row / full-tensor, Campaigns 11–12)
destroy exactly that arrangement while preserving statistics, so
an intact-vs-scrambled gap here would show the pipeline *can*
detect learned structure; continued absence would implicate the
surgery regime itself (frozen graft + small trained seams + tiny
budgets) rather than the cross-family pairing.

**Capability being tested:** general language/math development NLL
(the project's canonical endpoint). The probe measures both
parents natively on the dev splits before selection is finalized:
if the Instruct donor shows no measurable native advantage or
difference, NLL is explicitly a weak detector (per brief Part 4)
and the campaign tests *structural detectability* (intact vs its
own scrambles), which does not require the donor to be better —
only different in a learned way. The sector weight delta
(base vs Instruct, per-tensor RMS) is also measured by the probe;
a near-zero delta would invalidate the pair and stop the campaign
before preregistration.

## The stranger-check opt-in (protocol decision, recorded)

The pipeline deliberately rejects same-family parents
(`Recipe.validate`: "Cousin parents rejected…") and identical
tokenizer fingerprints (`load_parents`). For this campaign only,
an explicit field `Recipe.positive_control` (default `False`)
will: skip the cousin rejection, skip the distinct-tokenizer
requirement (recording `tokenizers_identical` instead), and make
`stranger_report()` report the truth (`different_architectures:
false`, mode labeled `same_family_positive_control`). The default
path is untouched — the existing cousin-rejection test must keep
passing, and every Campaign 13 run record will carry
`positive_control: true` in its structure/provenance so no reader
can mistake this for a cross-family result. This is the
"missing-rule" gap the brief's Part 5 anticipates: the canonical
protocol defines no positive-control rule, so one is preregistered
here and in `preregistration.md`.

## Candidates screened

| Candidate pair | Verdict | Reason |
|---|---|---|
| SmolLM-135M (recipient) × SmolLM-135M-Instruct (donor) | **selected** | Same config/tokenizer/lineage; instruct delta measurable; 135M keeps 20-run CPU matrix feasible (~10–16 min/run as in Campaign 12); all pipeline machinery (Llama blocks, seams, controls) applies unchanged. |
| SmolLM-360M × SmolLM-360M-Instruct | reserve | Same logic at 960 width; ~3× compute; held as fallback if the 135M pair fails validation for a size-specific reason (it is not expected to). |
| TinyLlama-1.1B intermediate × TinyLlama-1.1B-Chat-v1.0 | excluded | Documented fine-tune lineage, but 1.1B CPU training makes 20 matched runs infeasible in Actions job limits (~1 h+/run). |
| Qwen2.5-Math-1.5B × …-Instruct (already registered) | excluded | Strong math capability delta, but 1.5B scale is compute-infeasible for the 20-run matrix on CPU; also changes evaluation emphasis away from the canonical endpoint. |
| Mamba-130M × Mamba-370M (same family) | excluded | Different widths/depths (768 vs 1024…) and no fine-tune lineage; not genuinely compatible. |
| Identical checkpoint (SmolLM-135M × itself) | sanity only | Brief condition F: mechanical reconstruction check, not evidence. Included as a separately labeled registry, never as the positive-control result. |

## Risks and limitations (known in advance)

- Instruction tuning may change *style* more than dev-NLL
  capability; the primary endpoint may be insensitive (mitigated:
  the intact-vs-scramble contrast tests structure detectability
  directly, and native parent scores are reported).
- Base and Instruct are *not* independent models; a positive result
  shows the pipeline preserves structure in a compatible setting,
  **not** that cross-family transfer works (brief Part 10).
- Fine-tuning touched all layers, so the "donor capability" is not
  localized to the sector; the sector is a carrier of learned
  arrangement, which is what the scrambles ablate.
