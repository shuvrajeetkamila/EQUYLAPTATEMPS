# HETEROGENEOUS AI MODEL SURGERY — continuation audit, 2026-10-13

## Actual starting state (verified, not assumed)

- Repository `/home/user/EQUYLAPTATEMPS`, remote `https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git`.
- Current branch `arena/97efdf4a-equylaptatemps`; clean starting worktree.
- Actual starting HEAD **20f456b8e935540fe572ddb6bbd56a3cbf8f94a1**, not the user's historical b077573.
- Existing PR **#1 OPEN**, not merged/closed; same head/branch, base main. Latest push/PR Python3.10 and Python3.11 checks all SUCCESS. Opt-in weight jobs skipped as designed for ordinary commits.
- Public project is **HETEROGENEOUS AI MODEL SURGERY**. Repository/package names are existing compatibility identifiers; no project/repository restart.

The 106-test stage is historical. Independent execution at this starting HEAD reproduced **137 tests**, with lint/format/dependency checks passing. Existing dated 106/108/137 records are retained with their revision context, not overwritten as current results.

## A. Verified existing functionality

1. Whole native Mamba/Qwen2 sectors form one unconditional default child path, with full dense interface/norm parameters only trainable. Default inference does not query complete parent models or combine/rout their answers. Causal masks/config-derived RoPE and native Mamba sequence state handling are implemented; cache-free inference remains slow.
2. The existing work already went beyond random parents: genuine pinned `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1` and `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` ran on connected CPU Actions.
3. Completed run **38293921543**, source **95b2831**, includes both native parent baselines, two calibrated donor candidates, held-out child/no-training/ablation/random-interface records, original-parent frozen hashes, exact fresh-process reload and structural reverse surgery. This audit independently recomputed every metric summary and checked actual pinned dataset identities. It does not independently re-execute 3.6 GB parent downloads in this 3.85 GiB sandbox.
4. The real selected child has 217,553,920 parameters (215,187,712 frozen; 2,366,208 new). Selected Qwen [20,22) replaced Mamba [11,13) on development loss, not test score. The earlier real structural child used Qwen [13,15).
5. Published bridge-only checkpoints each contain exactly eight NEW tensor keys, 9,465,560 bytes, no frozen parent weights. Structural interface SHA `bcc43ee4fda51d35518fd79e0f27e23b4c503095dcde8320f4cdb06293f2a294`; selected-pilot interface SHA `5f2337211dd54ea0dd001244735cc3e7c3d2b74468eaea99053ad6a84411d537`. Parent references/tokenizer fingerprints/provenance are recorded. Full standalone child export/reload was measured remotely, not distributed.
6. The original synthetic ZIP is unchanged: 88,457 bytes, SHA `a3af87ff2cc0d35cbbf4e0dcf2b67460e2af9aeafc2762439675470b5a4d5be6`. Independent extraction/load/inference gave finite `[1,3,259]` logits with no parent directories. A first audit helper wrongly searched for `manifest.json`; the actual format is `child/child.json`. Correcting that helper did not require changing the repository archive.
7. Current negative science remains unchanged: math child 0/4; WikiText BPB child 3.444856383352949 versus Mamba 1.765828132006639 and Qwen 2.5806685715288404. Strict math score is zero for all. Flexible parent matches can be incomplete/intermediate numbers. Four math cases/32 generated tokens do not establish useful math ability. Inheritance criteria are NOT_RUN below the registered minimum64, never PASS.

## B. Independently reproduced weaknesses/failures

**Not new scientific results; controlled audit mutation tests only:**

- Changing a saved `inheritance.retention` summary to an invented PASS was accepted by `validate_results`, although the underlying unchanged pilot is below min64. The verifier recomputed benchmark scores but did not recompute inheritance verdicts. Historical science records were not modified.
- A bridge-only checkpoint checked tensor hashes but not execution settings. On tiny offline fixtures, changing the donor config's RoPE base preserved tensor hashes and was accepted by `load_reference`; the measured logits changed by max absolute 7.748603820800781e-7. This is a reproducibility weakness, not evidence of real-parent capability/performance.
- Existing sector inventories contain null/NOT_RUN selection fields even after the runner selected a candidate; search results elsewhere explain the selection but the inventory itself should reflect it.
- Current calibration/random-interface controls do not include training-budget-matched random sectors. Current generated-token records do not record stopping reason, so truncated reasoning can acquire a flexible numeric match. Both limitations are disclosed, not hidden successes.

The failed first real pilot (missing donor-tokenizer field) was already corrected and its partial records retained. This audit does not erase it or describe it as complete.

## C. Planned/unproven, not measured functionality

- Meaningful math/language retention, complementarity or improvement: **not established**.
- Registered larger 64-case/window GPU/Colab study: **NOT_RUN**; notebook/script existence is not execution.
- Multi-seed and matched-budget randomized-sector science: **NOT_RUN**.
- Validated integrated KV/SSM caching and native-precision GPU comparison: **NOT_RUN**.
- Reverse capability training/benchmarking: **NOT_RUN**; finite reverse forward is structural only.
- Direct Mamba converted-checkpoint license remains absent; official original checkpoint declares Apache2. This provenance licensing gap is preserved; no frozen weights redistributed.
- Checksums/registered configs detect inconsistency but cannot prevent a malicious author from forging an entire output bundle and recalculating all hashes.

## D. New audit work and necessary fixes (before additional experiment)

- Added portable v2 bridge execution contracts: normalized numerical config plus native module classes/operator settings (including norm epsilon and block indices), independent of cache path/installation metadata. New references reject altered RoPE/norm execution semantics, not only altered tensors.
- `load_parents` captures execution settings at verified load time. Legacy v1 real reference checkpoints require these fresh verified contracts; old published files are preserved, not rewritten. Reconstructing through the documented loader remains supported.
- Missing snapshot metadata now raises a deliberate validation failure rather than an incidental AttributeError.
- Result verification checks equal ordered benchmark identities across variants, registered sample counts, and recomputes registered inheritance verdicts. Unsupported edited PASS claims fail.
- Added six rejecting/portability regression tests. Independent after-fix suite: **143 passed**; all prior137 remain, lint/format and historical actual-record recomputation pass.

Evidence: `reports/audits/2026-10-13/{before-tests.xml,after-tests.xml,verification.json,resources.json}`. Test counts are measured execution, mutation outcomes are audit-only observations, real pilot scores are historical measured/recomputed data, unavailable studies are explicitly NOT_RUN.

## Resources and supported next work

Fresh sandbox inspection: 4,131,278,848 RAM bytes, no CUDA/GPU, 14,918,774,784 disk bytes free at check; Python3.11.2, torch2.5.1+cu124, transformers4.48.3, Hub0.36.2, safetensors0.8.0. Fresh Hub request returned TLS/SSL EOF; no gigabyte parent download attempted here. Metadata/source access and previously demonstrated connected CPU Actions are available. Complete independent integrity/report improvements and use a separately opted-in connected runner for genuine checkpoint reconstruction. Do not blindly expand the failed experiment or substitute another random-parent demo for science.

This audit and necessary fixes are committed/pushed **before any further real-parent execution** in this continuation.
