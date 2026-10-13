# HETEROGENEOUS AI MODEL SURGERY — Campaign 2 initial audit

Date: **2026-10-13**, Asia/Calcutta. This is a continuation of real pretrained experiments, not a restart from synthetic parents.

## Actual repository and workspace reconciliation

Remote `https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git`, branch `arena/97efdf4a-equylaptatemps`, PR#1 **OPEN**, verified remote starting commit **84205a51a44206f08a2f1c8219f0684fc1779626**. Latest Python3.10/3.11 CI green.

The local Git checkout initially pointed to `ecc8b0d` while its saved workspace contained older campaign files. Of 182 local files,178 matched the remote and four were older overlapping versions: README.md, docs/pretrained-study.md, scripts/publish_study_evidence.py, src/equylaptatemps/evidence.py. No only-local source files were found. All prior local work was preserved in a Git stash (SHA recorded in initial-state JSON); the same branch was fast-forwarded to84205a5 without reset, force-push, switching branches or deleting the root/Git directory. The preserved stash is not dropped or silently applied over newer source. This avoids interpreting a stale workspace as the latest experiment.

## A. Independently verified existing functionality

- **197 tests passed**, lint/format/dependency checks passed in newly prepared local environment. Historical106/158/etc reports remain revision-specific subsets, not current test counts.
- Existing default child is one integrated native-block path, Mamba prefix→full dense/norm→whole Qwen sector→full dense/norm→Mamba suffix/head. Neither complete-parent answer queries, router, averaging, LoRA nor distillation is used.
- Published selected child:217,553,920 unique parameters;215,187,712 frozen;2,366,208 new interface/norm parameters. Whole Qwen[20,22) replaces Mamba[11,13). Current bridge has per-token LayerNorm and unfactorized linear768↔1536 matrices; an identity is not valid at unequal dimensions.
- Actual parent checkpoints remain `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1` and `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`, native selective SSM/convolution versus GQA/RoPE transformer. Parent configuration/tokenizer/weight hashes and notices are present. Direct converted-Mamba license-document gap remains; official original Mamba declares Apache2. No frozen weight distribution.
- Original synthetic ZIP and both actual new-interface binaries are byte-unchanged; hashes are in the initial state record. They are preserved, not reclassified as useful pretrained models.
- The completed64-window language records are independently recomputed against newly fetched immutable real dataset blobs. Baseline BPB: Mamba1.7921447898729175;Qwen2.34309137844324;child3.2795770048682704;matched random donor3.043678304653564. Lower is better. Same examples/byte count, native independent tokenizers, BOS context/EOS target; canonical BPE token sequence likelihood, not word-level perplexity.
- The latest longer math run **38310648576**, source **c271693d833521b44bd8d53776dde78a547b64c3**, completed but its scores had not yet been incorporated into the remote pending report. The actual source-bound bot gzip/base64 bundle was retrieved, SHA-verified and imported under `baselines/math_run_38310648576/`, then scores, exact dataset questions/gold/prompts/selection/token budgets were recomputed locally. **Qwen12/16 completed-answer correct;child0/16;Mamba0/16.** Two human TRAIN fewshots, fresh TEST questions,256 output tokens, explicit final-marker scoring, no arbitrary-last-number fallback. This is a16-case diagnostic, not full64/multiseed inheritance.
- Native CPUfloat32 request-local KV/SSM caching was added in prior campaign, retaining original sparse block indices/global causal positions and full-prefix numerical oracle. The zero-input cached control failed a real oracle gate and is explicitly blocked; tolerance was not relaxed. Failed run38307630994 preserved. Other variants passed real gates; native-parent/child/reverse/empty-sector deterministic unit tests remain.
- Full private child-only checkpoint save/fresh process generation, verified parent-reference reconstruction and frozen invariance are recorded. Published files are reference-dependent, not standalone; no claim that current local4GiB machine reloaded real parents.

## B. Failures and limitations

Current child executes but generates repetitive punctuation/words. Registered language retention and learned-sector advantage FAIL. The matched random sector performs better. Math donor demonstrates a useful diagnostic capability that the child does not retain. Ablation degradation is not inherited usefulness.

Tensor execution, file integrity and primary causal masks/recurrence are tested, but diagnostic caches are CPUfloat32/batch1/no padding only. One cached ablation remains unvalidated/blocked. Larger GPU/Colab, additional general reasoning, confirmatory multiseed and full64-case long math not run. Frozen record checks are not authentication against maliciously forged complete bundles.

No new source regression was found in the197-test starting suite. Records are reverified; actual fresh tensor-level baseline reproduction in **Campaign2** remains NOT_RUN at this audit and must be gated before promotion. Native/model hidden statistics have not yet been measured in this campaign. Previous estimated residual-scale mismatch is a hypothesis, not a proven cause.

## C. Resources and feasibility

Current sandbox:4,131,278,848 RAM bytes,noGPU/CUDA,~11.9GB free after dependency installation; Python3.11.2,torch2.5.1+cu124,transformers4.48.3,Hub0.36.2,safetensors0.8.0. Fresh Hub API request failed TLS/SSL EOF. Real parent downloads/loads are not attempted locally. Dependencies were installed from permitted PyPI; CUDA wheel libraries exist but no CUDA hardware. Connected CPU GitHub Actions has already loaded exact parents with ~16GiB RAM; use explicit opt-in bounded sequential jobs and reuse snapshots within a job. No six-hour/background-after-turn guarantee is made.

## D. Campaign 2 decisions before new experiments

1. Freeze/import/recompute the actual negative language and completed math baseline; rerun original parents/child/matched random control on a connected runner where possible.
2. Diagnose input/output residual statistics (means, variances, covariance, norm/position/extreme/nearzero/finite rates), native operation semantics and new-interface gradients/updates before bigger topology/training changes.
3. Use a small preregistered alignment matrix. Distribution-aware initialization may use unpaired aggregate unlabeled statistics, but **no hidden-activation vectors, parent distributions or generated answers become training targets**. Direct child next-token/human TRAIN objective only; all original tensors remain frozen.
4. Primary controls preserve budgets/seeds/data/dtypes and stock block operations; identity at unequal widths is explicitly invalid. Alternative one-block/early/reverse paths receive cheap screening, not blind full training.
5. Add resumable safe new-parameter/optimizer checkpoints and immutable registry/state/config/artifact/report validation. Skip only verified completions; preserve interrupted/failed records; stop at recorded wall-clock/disk/memory limits.
6. Promote only on development data under predeclared gates; final held-out results are not used to repeatedly select the bridge. Both language AND math matter. Outcomes A/B/C/D remain evidence-driven.

Machine-readable state, current baseline hashes and raw starting JUnit are beside this report. This audit is committed/pushed before Campaign2 real-parent work. Model binaries/history/earlier reports are not overwritten.
