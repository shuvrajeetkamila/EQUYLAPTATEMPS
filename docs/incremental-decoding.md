# HETEROGENEOUS AI MODEL SURGERY — native incremental decoding

## Purpose and boundaries

Longer math generation was impractical with repeated full-prefix execution. The new
`decoding.CachedDecoder` is an inference implementation for the **same integrated
child**, not a new model or capability claim. Original cache-free `SurgicalChild.forward`
remains the default/training path and the numerical oracle. No weights, parent block
indices, normalization, RoPE configurations, embeddings or heads are changed.

Initially supported: **CPU float32, batch1, no padding**. CUDA/BF16/multi-batch cache
validation is NOT_RUN and rejected by this API, rather than silently assumed equivalent.
Existing un-cached GPU/BF16 paths remain available under their earlier documented limits.

One request gets separate native cache objects for recipient and donor. Recipient
prefix and suffix share its original-layer-indexed cache; donor states remain separate.
Mamba stores causal convolution history and selective SSM state. Qwen stores its native
keys/values and uses explicit global positions, config-derived FP32 RoPE and causal
masking. Sparse original layer indices remain unchanged; omitted parent blocks are
not executed. No full parent-model inference is called by the child's path.

Caches are transient activations, outside the model state dict/checkpoint. They must not
be reused between prompts/models/control modes. Prefill happens once, then one token per
step; model training/device/dtype/parameter changes fail. Training always executes the
original full-sequence implementation. The parent cache decoder is used only for separate
baseline measurements, never an ensemble or task router.

## Validation and known restriction

Offline native fixtures check full-prefix/cached logits for both architecture directions,
native standalone parents, original-index gaps, empty prefix/suffix, supported ablations,
request isolation and unchanged frozen/model tensors. These tests prove implementation
mechanics, not pretrained capability.

First real-weight run **38307630994**, source **1d2a960**, passed cached-vs-original checks
for both exact pretrained parents, the published child, untrained/random interfaces,
matched random sector, donor bypass and prefix/suffix bypass. All checked greedy argmax
choices matched. Maximum absolute observed FP32 differences for those passing cases were
about0.00008–0.00042 (actual per-case records retained).

**Zero-input-interface cached equivalence failed** the same numerical gate before any
benchmark TEST answer was scored. That failed run is preserved. The API now explicitly
rejects this cached control; its original uncached implementation and earlier real
ablation records remain intact. The tolerance was NOT loosened. No failed oracle is
called PASS. The restricted retry registration excludes only this cached control before
new test observations; other recipient/interface ablations remain registered.

CPU matrix shapes can change floating rounding. Real experiments must pass the registered
logit tolerance **and identical greedy decisions** on development/training prefixes before
benchmarking. They stop on a failure instead of reporting scores for an altered model.
This limited oracle is not exhaustive numerical proof for every possible sequence.

## Completion-aware math diagnostic

Registrations:

- Original attempted: `configs/pretrained-completed-math.json` (failed oracle run preserved).
- Restricted validated scope: `configs/pretrained-completed-math-validated.json`.

The bounded CPU diagnostic uses16 fresh GSM8K TEST cases excluding the previously inspected
four, two **human GSM8K TRAIN** few-shot examples outside prior calibration/development,
up to256 new tokens, and identical prompts/precision for both parents and unchanged child.
No new training of the published child occurs. The matched random-sector control alone
repeats the existing identical16-step/data/seed/interface budget; it is explicitly CONTROL,
never called pretrained inheritance.

Primary scoring requires a **completed `####` numeric line** or closed numeric `boxed`
scalar. A trailing digit at a token-budget cut does not count; arbitrary intermediate
numbers never receive last-number fallback credit. Records include every generated ID,
raw output (line endings preserved), prompt/question/gold identities, stopping reason,
sample count, Wilson interval, runtime and failure. All failed/truncated samples stay in
the denominator. Different prompting/output/scoring means this is not the unchanged
4-case pilot or a comparable published leaderboard score.

Original inheritance thresholds are NOT lowered: minimum64 math cases/multiple-seed
retention and previously failed language evidence remain separate. This diagnostic can
clarify actual completed math answers but cannot establish dual-capability inheritance.

```bash
# Connected >=12GiB RAM/free disk CPU machine, repository installed:
python scripts/run_completed_math.py --config configs/pretrained-completed-math-validated.json --out artifacts/completed-math-002
python scripts/verify_study.py artifacts/completed-math-002 --datasets data/local/pretrained-benchmarks
```

Standard CI is offline/lightweight; `[completed-math]` explicitly opts into the separate
connected weight workflow. Always use a new output directory and retain failed attempts.
No parent weights/raw large datasets/duplicate interface files are committed.

```python
from equylaptatemps.decoding import cached_generate
result = cached_generate(child.eval(), tokenizer, prompt,
                         max_input_tokens=2048, max_new_tokens=256)
print(result.text, result.stop_reason)
```

Plain execution/generation is not useful inherited capability. Negative outputs and gate
failures must remain reported. GPU cache support, exhaustive long-sequence equivalence and
useful calibrated semantic alignment remain research work, not hidden completion claims.
