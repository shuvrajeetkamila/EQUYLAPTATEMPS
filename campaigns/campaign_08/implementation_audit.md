# Campaign 8 Implementation Audit — Campaign 7 Forward Path (verified from source)

Every claim cites the file or record it was verified in. Source is authoritative
over any summary.

## Pairing and topology (unchanged for Campaign 8)

| Fact | Value | Source |
|---|---|---|
| Host | `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | `campaigns/campaign_07/experiment_registry.json` |
| Donor | `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4` (Llama, width 576, 30 layers, GQA 9/3, tied embeddings) | same; probe evidence PR #1 |
| Recipient cut / retained host blocks | cut [11,13): prefix [0,11), suffix [13,24); 2 host blocks removed | registry; `model.py::child_from_parents` |
| Donor sector | contiguous **[24,30)**, native order, 6 blocks, ≈23.6M params | registry; `backbones.py::extract_parent` |
| Child | 143,721,408 unique params; context limit 2048 | `campaigns/campaign_07/screen_38542143640/runs/c07_dense_pre_42/results.json` structure |
| Interfaces | two `DenseSeam`s (LayerNorm + full Linear 768→576, 576→768), **888,768 trainable params**, the only trainables | `model.py::DenseSeam`, `assert_trainable_contract`; results.json |
| Campaign 7 initialization | `random` (DenseSeam defaults under recipe seed) | registry experiments; `alignment.py::initialize_conditioned` |
| Donor internals | frozen; fingerprints verified before/after in runner; RoPE via `StackSegment.rotary` (Llama branch), eager attention, additive causal mask | `backbones.py`, `campaign07_runner.py` |
| Random twin | same-shape native Llama blocks, `donor._init_weights`, `initialization_seed=1234`, frozen | `controls.py::random_sector_control` |
| Tokenization | host (Mamba) tokenizer end-to-end for the child; donor tokenizer only for donor-native references | `campaign07_runner.py`, `model.py` forward vocab guard |
| Metric | macro = mean(language, math) development token NLL; dev = WikiText-2 validation 16 windows + GSM8K dev 16 (host tokenizer) | `campaign03_screen.py::macro`, runner |
| Gates (C7, reused) | G-B: macro_pre ≤ 0.98 × macro_twin; G-C: language ≤ 1.50 × host (4.5595) and macro ≤ 6.00 | `campaigns/campaign_07/preregistration.md` |

## Verified Campaign 7 results (Stage A baseline, reused — not rerun)

From the SHA-256-verified import (`campaign_07_results.json`, bundles
`d568e599…` / `3c59160e…`): seed 42 pretrained **8.5719** vs twin **7.5349**;
seed 123 pretrained **9.4023** vs twin **7.8900**. Re-running them would spend
compute to re-derive stored, test-traced numbers, so Campaign 8 adopts them as
Conditions 1 and 3 and records that decision here and in the preregistration.

## Calibration machinery available (existing, tested)

`alignment.py::initialize_conditioned` supports `covariance_orthogonal`:
variance-preserving orthogonal seam pair + input-seam whitening by the observed
child-input covariance + residual-RMS scale matching, all estimated from
unlabeled TRAIN text by `campaign02_align.py::measure_initial`. No parent
hidden activations are used as targets. The Campaign 7 runner already calls it
per-experiment from the registry `initialization` field, and measures
per-child statistics first (so a random twin receives the identical estimation
procedure on its own child). This is the Campaign 8 calibration method.

## Discrepancies vs the campaign brief's assumptions

None material: the six-block donor sector [24,30), host, revisions, seeds
(42/123), twin seed (1234), and gates all match the brief. One clarification:
the brief's suggested branch `arena/campaign-08-interface-calibration` cannot
be created in this session (branch is fixed to
`arena/97efdf4a-equylaptatemps`, PR #1); work proceeds there, as in Campaign 7.
