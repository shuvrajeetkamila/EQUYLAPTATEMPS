# HETEROGENEOUS AI MODEL SURGERY — completed-answer math diagnostic

**Date:** 2026-10-13 (Asia/Calcutta). **Purpose:** address the invalidity/underpower of the earlier four-case32-token math pilot without changing frozen pretrained parents, child parameters, excluded methods or inheritance thresholds.

## Verified continuation, audit before experiment

Actual starting repository/branch/PR was **ba938989f3d24bfce40508d32cd962832a7d6689**, not the historical b077573 identifier. PR#1 OPEN, Python3.10/3.11 checks green,158 tests independently reproduced. Original synthetic ZIP and both actual real NEW-interface binaries/checksums unchanged. Historical real math/language/control records independently recomputed against pinned dataset identities. Source/audit report and raw JUnit/resources were committed/pushed **b714373bd3ce1c0b221234a6ab1897da69768317 before any new real-weight experiment**.

Sandbox remains~3.85GiB RAM/noGPU/HubTLS-EOF. Genuine weights are acquired only on the separately opted-in connected CPU runner, at common float32 with verified exact model IDs/revisions/safetensor/config/tokenizer hashes. No parent weights, huge dataset copies or duplicate interface binary committed. Existing cached parent snapshots are reused within each run. GPU cache validation remains NOT_RUN.

## Exact models and unchanged child

- Recipient `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`,129,135,360 original parameters, selective SSM/convolution,width768/24 blocks, native Mamba tokenizer/vocabulary50280.
- Donor `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`,1,543,714,304 original parameters, GQA/RoPE causal transformer,width1536/28 blocks,vocabulary151936.
- Exact whole learned Qwen[20,22) replaces Mamba[11,13), one integrated Mamba prefix → dense/norm → native Qwen sector → dense/norm → Mamba suffix/norm/head. No donor embedding/head or averaged weights, no routing/two-parent answer combination.
- Published child remains **217,553,920 total /215,187,712 frozen /2,366,208 new interface parameters**. It is not retrained in this diagnostic. Existing new-interface checkpoint SHA `5f2337211dd54ea0dd001244735cc3e7c3d2b74468eaea99053ad6a84411d537` is reused exactly.
- Qwen direct snapshot Apache2; Mamba conversion direct license gap remains, with official original checkpoint Apache2 lineage evidence disclosed. No frozen tensor redistribution or implied model-license grant. DatasetGSM8K is MIT; human-written train/test only, never `example_model_solutions` teacher data.

## Native incremental execution is not a new architecture or scientific result

`decoding.CachedDecoder` keeps request-local native Mamba convolution/SSM states and Qwen KV state, explicit global RoPE/causal positions, and original source layer indices including gaps. Recipient prefix/suffix share their native cache; donor state is separate. They remain the same one-path child, not full parent inference/answer routing. Caches are transient activation state, not checkpoint/model tensors.

Only **CPUfloat32,batch1,no padding** supported. Training retains the original cache-free path. No config or pretrained tensor value changes. Model/control binding is immutable during a request; training/device/dtype/parameter changes fail; caches cannot be reused across prompts. Both architecture directions, native parents, empty prefix/suffix and sparse source indices tested with original full-prefix numerical oracle. Those offline tiny fixtures are engineering evidence only, not pretrained scores.

Real evaluation must independently pass the registered cached-versus-full-prefix tolerance0.0005absolute/relative **and identical greedy argmax decisions** on real source TRAIN prefixes before scoring TEST questions. This limited gate is not exhaustive numerical proof for all sequences. Floating matrix-shape rounding is disclosed rather than claimed bit-identical.

## First actual attempt — failed gate, preserved, no benchmark scored

Connected run **38307630994**, source **1d2a960e24dd00c8b00b99dac9c2f01f9d982d0e**, loaded the exact original models/real child and repeated the matched random donor-control budget. Runtime **494.142584474 seconds** was measured before abort.

Nine passing oracle records were written: both native parents, published real child, uncalibrated/random interfaces, matched frozen random donor, donor bypass and prefix/suffix bypass. Greedy choices matched in every passing case. Actual max absolute differences ranged about0.00008–0.00042; precise values/timings in saved JSON.

**Zero-input-interface cached equivalence failed** the same numerical gate. The script stopped before any test answer was scored; `comparisons` remained empty, not a zero benchmark score. Failure/error/source/provenance/control-calibration/data-selection/registration records retained under `reports/pretrained/attempt-38307630994/`. This is a failed inference-control validation, not failed parent mathematical capability.

No tolerance or inherited-capability threshold was relaxed. The API now explicitly rejects this unvalidated cached control; its original uncached implementation and all earlier actual ablation records remain available. A separate registration excludes it from the cached longer-output scope before final test observations. Other recipient/interface and donor controls remain. New rejection tests supplement rather than weaken the original uncached regression cases.

## Restricted retry registration

Config `configs/pretrained-completed-math-validated.json` was committed at **c271693d833521b44bd8d53776dde78a547b64c3** before the retry's benchmark outputs. Connected retry **38310648576** was launched; results are included only once its externally source-bound raw bundle is received, checksummed and recomputed.

Protocol:

- Original immutableGSM8K TEST; seeded16 questions excluding all four previously inspected test IDs588/1214/57/906.
- Two HUMAN GSM8K TRAIN few-shot examples outside prior calibration/development. Same text/prompt instructions across models; each uses its own native tokenizer. Train/test prompt overlap checked. No teacher model or test-target training.
- Greedy native cached CPUfloat32 generation, up to **256 new tokens**,maxinput2048/context<=4096. No silent prompt/reference truncation.
- Stop on EOS, completed `####` numeric line or closed numeric boxed scalar, otherwise token budget. Preserve raw line endings, every generatedID,inputcount,terminationreason,prompt/question/gold hashes,time,failures.
- Primary numeric equality requires a **completed final marker**, never arbitrary last/intermediate numbers. A token-budget cut at a trailing digit does not count as complete. Failed/truncated outputs stay in denominator; Wilson95 intervals and completion counts reported.
- Baselines/control variants: both exact original parents, unchanged calibrated child, untrained child, independent random interfaces, matched frozen random donor, donor bypass,prefix/suffix bypass,zero-output interface. Zero-input cached variant explicitly BLOCKED, not secretly replaced with successful scores.
- Matched random donor is CONTROL only; standard native initialization seed1042 on two frozen Qwen blocks, same original16humanGSMTRAIN+8WikiTRAIN,seamseed42/16steps/LR0.0003/batch1/AdamW/checkpointing/trainable scopes. No claimed random-weight inheritance.
- Reverse cache correctness checked cheaply; reverse capability training/scaling NOT_RUN. Published child not newly trained. Checkpoint export/fresh child-only cached generation with original parent loading forbidden checked before completion.

This is a different prompting/length/scoring procedure, not an unchanged published leaderboard result or retroactive improvement of the earlier four-case score. Original inheritance minimum64/multiple-seed/source and language-retention requirements remain: dual-capability inheritance is NOT_RUN/unproven in this bounded16-case math diagnostic. The earlier64-window language retention and matched-sector advantage FAIL records remain unchanged.

## Result status and interpretation

At report initialization, retry completion/scores are pending and unavailable—not assumed or zero. Actual results will be appended after independent verification. Negative outputs, incomplete answers and cache-gate failures remain visible. No skill inferred from model names or architecture.

## Reproducibility and tests

Ordinary CI remains small/offline; real weights are opt-in `[completed-math]` only. Offline source/scorer/cache/request-isolation/schema tests pass; exact final count/checks listed in current validation record/PR. Existing source/archive tests unchanged. Metric/config/raw-record/registered-sample/final-marker/frozen-tensor/fresh-process checks reject missing/manually substituted/inconsistent records. Checksums cannot authenticate a maliciously fabricated entire bundle.

```bash
# Connected >=12GiB RAM/free disk CPU machine, repository installed:
python scripts/run_completed_math.py --config configs/pretrained-completed-math-validated.json --out artifacts/completed-math-002
python scripts/verify_study.py artifacts/completed-math-002 --datasets data/local/pretrained-benchmarks
```

See `docs/incremental-decoding.md` for decoder contracts/blockedcontrol and `docs/pretrained-study.md` /existingColab notebook for the larger GPU study, whose execution remains NOT_RUN. Original pretrained/interface data never silently quantized or merged. No LoRA/distillation/weight-averaging/answer-ensemble/router introduced.

## Next research decision

Use completed-answer parent/math/control measurements together with the existing failed language/matched-random evidence to decide whether the current donor meaningfully contributes. If the child still fails, prioritize development-only hidden/residual/token-semantic alignment across seeds before larger/reverse sector scaling. A usable, retained dual-task model—not a faster forward pass—is the research target.
