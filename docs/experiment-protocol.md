# Pretrained experiment protocol

The implementation is runnable, but no real pretrained transplant was validated in the original sandbox. Follow this protocol before reporting a trained “child.”

## 1. Record actual parent provenance and rights

Use independently pretrained Mamba and Qwen2 checkpoints from the reviewed registry. Record immutable commits, source model cards, licenses, actual architectures, vocabulary contracts, and hashes. `scripts/prepare_parents.py` can acquire/pin snapshots on a machine with Hugging Face access. It downloads safetensors only and verifies available Hub LFS SHA-256 metadata.

`build` loads locally with `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`. Available download inventories are verified before model loading. Parent LICENSE/NOTICE files found in local snapshots are retained in exported artifacts, with hashes. Missing license documents are reported, not magically replaced by this repository's software license. Determine redistribution rights yourself.

The recipe template has intentionally invalid revision placeholders. Never replace them with fabricated hashes or `main`. Synthetic parents remain explicitly labeled synthetic, including when loaded through the same local-snapshot path.

## 2. Plan enough hardware

Start with the 130M/1.5B recipe, not the largest registry models. Loading keeps both full parents in memory temporarily; search keeps full parents alive for reuse. During seam training, autograd retains/recomputes operations through frozen graft and suffix. Attention, recurrent intermediates, and vocabulary logits can dominate memory despite a small optimizer.

Choose a common dtype and one device. Test BF16 support on your GPU before a long run; CPU float32 and the stock sequential Mamba backend are useful for small validation tests. No GPU/optimized-kernel result is included in the checked-in evidence. Use short sequences and gradient checkpointing initially. There is no automatic VRAM sizing, sharding, quantization, or cache optimization.

Exported files can contain hundreds of millions of retained parameters. Keep snapshots/checkpoints outside ordinary Git, using only storage permitted by parent licenses.

## 3. Curate three splits

Prepare **train**, **validation**, and **final test** JSONL with known-correct answers, not parent-generated teacher outputs. Include both desired specialist domains and recipient-retention tasks.

- Math: held-out multi-step questions, not just arithmetic formatting.
- Reasoning: valid problems with independently checked answers.
- Code: correct reference outputs plus separate executable task tests; strict answer strings alone are inadequate.
- Recipient retention: language/general tasks and whatever qualities motivated the recipient choice.

Respect dataset licenses. Prefer recognized benchmarks and task-appropriate scoring, with documented contamination controls. The shipped examples are hand-authored schema illustrations, not benchmark datasets.

The implementation rejects normalized identical prompts between training and validation, but cannot catch paraphrases, benchmark contamination in parent pretraining, or semantic leakage. Record hashes and versions of all splits; never publish private data unintentionally.

## 4. Obtain baselines before calibration

Evaluate the intact recipient and donor using their own tokenizers on the same final tasks. Save task scores, latency, context conditions, and failure examples. Child/recipient response-token NLL can be compared directly because the child's vocabulary is the recipient's. Donor-token perplexity is not comparable across different tokenizations.

Plain completion prompting is supported (`prompt + "\n\nAnswer:"`), not native chat templating. Report this limitation, especially with instruction checkpoints; do not confuse a prompt-format failure with absence of a parent capability. For a publication-quality comparison, supply fair parent-specific prompting/evaluation outside the smoke CLI as well.

## 5. Choose and calibrate whole sectors

Do not assert particular layers contain isolated “best reasoning code.” Select a recipient cut that leaves meaningful native prefix/suffix computation, then test multiple whole donor windows with `search`.

Each window gets independently initialized dense seams at the same seed and calibration budget. Only new seam parameters are optimized. Rank on validation data, track per-domain retention, and keep the final test set sealed. The built-in search holds recipient cut and donor-window length fixed; it does not perform exhaustive architecture search or infer functional organs.

Report failed windows too. A window winning a small search is only best *among those tested under that objective/budget*.

## 6. Verify integrity and necessity

After calibration:

- Confirm source graft and all frozen tensor hashes are unchanged at the chosen loading precision.
- Check the only trainable parameters are the two dense seams and their normalizers.
- Save/load the child without parent directories and check logits.
- Run full-child, `bypass_graft`, and `zero_graft` ablations. A near-zero bypass penalty weakens any claim that native donor computations are carrying the specialist skill.
- Compare with the untouched recipient and donor on an independent test set.
- Repeat with multiple seam seeds, a randomly initialized sector control, and recipient-only continued supervised training with a comparable parameter/compute budget in a separate experiment.

The frozen-tensor audits show the implementation's mechanics, not whether the child acquired useful reasoning. A loss decrease can result entirely from fitting seam parameters.

## 7. Report measured results, not inherited labels

At minimum report task-specific accuracy/pass rates, domain retention, uncertainty across seeds, response-token loss, model/bridge sizes, hashes, precision, hardware, prompting, context lengths, calibration examples/steps, and inference cost. Use robust numerical/semantic math grading and code tests in an appropriately sandboxed evaluator; the CLI's exact-match metric is deliberately basic.

Do not claim transfer unless the child's task scores support it and controls establish the donor sector matters. Do not claim “best of both” if recipient retention fails. Do not claim Mamba speed/linear attention complexity once a full-attention donor is inserted. Distinguish an initialized splice, calibrated interfaces, and a genuinely validated model.

## Checked-in evidence

Only synthetic CPU evidence is included:

- Native Mamba/Qwen2 execution agrees with decomposed sector execution.
- Both splice directions, causality, masking, gradient checkpointing, immutable parent tensors, and portable serialization are tested.
- The demo calibrates a tiny random-parent child and measures losses/ablations honestly.
- No pretrained weight acquisition, GPU training, foundation-model benchmark, or successful math/reasoning inheritance is asserted.
