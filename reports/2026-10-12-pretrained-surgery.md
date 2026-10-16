# HETEROGENEOUS AI MODEL SURGERY — real pretrained transplantation

**Research date:** 2026-10-12. **Scope:** real whole-block transplantation and bounded controlled evaluation, not a claim of successful inheritance.

## Executive finding

A genuine pretrained stranger-model child **was built, executed, calibrated on real text, saved, and reloaded in a fresh process**. This is a material advance over the preserved random-parent prototype. The first measured integrated child has **217,553,920 parameters**, of which **215,187,712 are frozen pretrained tensors** and **2,366,208 are newly initialized full-sized interfaces/norms**. Neither LoRA, distillation, averaging, ensembles, nor answer/task routing was used.

**Useful capability inheritance remains unproven.** Eight calibration steps reduced a tiny real-text fitting loss, but the short generation sample was `io`. This is not a usable-language or mathematical-reasoning result. Benchmark test targets were not used for calibration. The completed controlled CPU pilot below produced negative capability results; no unexecuted metric is represented as zero.

## A. Independent audit and corrections — completed before real-parent work

Audit commit **7d968086a0f2a4f03a9f57bf630dbebd59bc405b** independently reproduced the original 106 tests and published synthetic ZIP extraction/reload/inference. A provenance weakness accepted caller-declared pretrained identifiers around random byte-tokenizer fixtures. The loader now requires verified pinned snapshot inventories for pretrained IDs and rejects byte fixtures. Two rejecting tests raised the suite to 108. Audit/fixes were committed and pushed before model acquisition.

The public project title is HETEROGENEOUS AI MODEL SURGERY. Repository/package/CLI identifiers are compatibility names, not a different project. The original synthetic archive is preserved: **88,457 bytes**, SHA-256 `a3af87ff2cc0d35cbbf4e0dcf2b67460e2af9aeafc2762439675470b5a4d5be6`. Its historical synthetic scores are engineering evidence only.

Audit details: `reports/audits/2026-10-12-initial-audit.md`, JUnit/resource/checkpoint records beside it.

## B. Verified actual parents and resources

| Fact | Mamba recipient | Qwen donor | Evidence type |
|---|---|---|---|
| Model ID | `state-spaces/mamba-130m-hf` | `Qwen/Qwen2.5-Math-1.5B` | remote API and downloaded snapshot |
| Immutable revision | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` | remote metadata/verified local record |
| Native architecture | selective SSM/causal convolution | GQA causal transformer/RoPE/MLP | actual config/classes |
| Original blocks / width | 24 / 768 | 28 / 1536 | measured config |
| Original unique parameters | 129,135,360 | 1,543,714,304 | loaded model count |
| Vocabulary dimension | 50,280 | 151,936 | actual config |
| Downloaded weight SHA-256 | `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea` | `f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b` | verified snapshot bytes / official LFS metadata |
| Direct converted snapshot license | **not supplied** | Apache-2.0 LICENSE/tag | observed, not guessed |

The official original Mamba checkpoint `state-spaces/mamba-130m`, immutable revision `c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0`, declares Apache-2.0. This is lineage license evidence; the converted HF snapshot still has a direct-document/tag gap. No frozen model weights are distributed in Git/artifacts. New interface weights and reproducible immutable parent references are provided instead.

These are computationally different pretrained architecture families, not same-family fine-tunes. Distinct tokenizer contracts are checked. Declared lineage roots are disjoint; hidden ancestry and shared/pretraining data cannot be excluded by an allowlist/checksum.

Sandbox: 4,131,278,848 RAM bytes, no CUDA/GPU, Hub TLS EOF, ~15 GB disk free at audit. It did NOT load the real weights. A connected GitHub CPU runner supplied **16,765,415,424 RAM bytes**, common float32, torch 2.5.1+cpu, two CPU threads, no GPU. Qwen's stored BF16 values were exactly promoted; execution rounding differs from BF16 published scores, so baselines use the same float32 conditions. No quantization or unsafe pickle conversion occurred.

Metadata-only connected run **38287990737** at **13e2a000278b517d81e9143d68050ebb8e7e4661** preceded weights. Its immutable evidence is `reports/audits/parent-metadata-remote.json`; that run itself did not download weights or perform inference.

## C. Actual transplanted topology and discovery scope

```text
Mamba tokenizer/embedding
  → complete original Mamba blocks [0,11)
  → NEW LayerNorm + full dense 768→1536
  → complete original Qwen blocks [13,15)
  → NEW LayerNorm + full dense 1536→768
  → complete original Mamba blocks [13,24)
  → original Mamba norm/tied output head
```

Mamba [11,13) is physically absent; Qwen embedding/head are not part of the child. Norms, native residual operations, attention/MLP/convolution/SSM internals and dependency handling remain intact. Native Qwen positions use configuration-derived FP32 RoPE and causal masks on the recipient token timeline. Native Mamba state starts afresh for each full sequence; no cross-request recurrence/cache is retained. This is one integrated unconditional default path, not model-output blending.

`examples/pretrained-structural-reference/sector-inventory.json` records actual tensor shapes, counts, dependencies and frozen source hashes. The first structural selection [13,15) was a compatible **hypothesis**, not a discovered math organ. Missing baseline/activation observations in that initial inventory are null, explicitly unobserved.

The controlled study adds development activation means/RMS/residual change, native-parent sector identity ablations, candidate calibration/ranking and compatibility/rejection records. It searches [13,15), [20,22) in the CPU preset; larger registered preset also searches [8,10). Selection uses only development NLL, not held-out test outcomes. Original source methods/models remain available for comparisons. Semantics of heterogeneous tokenizer/hidden spaces remain a learned alignment problem.

## D–F. Real execution, permitted calibration and reload — measured

Connected run **38290378364**, source **f812e6b5d74e7465e02cd6f41df71589dc35ad60**, completed:

| Observation | Value | Kind / interpretation |
|---|---:|---|
| Integrated unique parameters | 217,553,920 | measured count |
| Frozen parameters | 215,187,712 | measured count |
| New trainable parameters | 2,366,208 | measured count; eight interface/norm tensors |
| Prefix / donor / suffix block calls in one default forward | 11 / 2 / 11 | measured complete execution trace |
| Finite forward-logit shape | `[1,4,50280]` | measured, not task capability |
| Calibration steps / train windows / dev windows | 8 / 4 / 4 | actual run; 64 characters/window |
| Initial train token NLL | 13.131952429330477 | measured tiny fitting diagnostic |
| Final train token NLL | 7.0268093642368115 | measured tiny fitting diagnostic |
| Development token NLL | 8.35311758858817 | measured small validation diagnostic; not benchmark |
| Calibration time | 6.440556681 seconds | measured wall-clock |
| Total probe time | 66.682520829 seconds | measured wall-clock, scope includes acquisition/checks |
| Fresh-process logits | bit-identical | measured; parent loading explicitly forbidden |
| Both original-parent and retained-child frozen tensors | unchanged | before/after hashes measured |
| Short generation sample | `io` | measured sample, quality unvalidated |

Objective: real WikiText TRAIN next-token cross entropy, including EOS; initial token masked. AdamW LR 0.001, betas [0.9,0.999], epsilon 1e-8, weight decay 0, batch 1, seed 42, max sequence 64, gradient checkpointing. All pretrained tensors were frozen initially; only full-sized new interface/norm parameters had gradients/optimizer membership. No parent outputs/activations were targets, no low-rank modifications, no averaging or routing.

Frozen-child SHA: `e758c9a5b1797652fd39d39c5200cfe8d46dbe0301f121513609d2a16a3eabf3`.
Graft SHA: `e61216ee63ee7fece28e8d6c88402d806bdce70d94973c7b1b202f459ae71458`.
Fresh/calibrated-logit SHA: `b2466d3aea07020fd5f74df6030e09efe791f5a57cd8b1ba814431a3ac7a5456`.

All values trace to `reports/pretrained/2026-10-12-cpu-structural.json`, public Actions run/comment, `examples/pretrained-structural-reference/calibration-report.json`, inventory and checkpoint descriptor.

### Actual reproducible child files

`examples/pretrained-structural-reference/interfaces.safetensors` is the actual trained **9,465,560-byte** new-interface file, SHA-256 `bcc43ee4fda51d35518fd79e0f27e23b4c503095dcde8320f4cdb06293f2a294`. It was retrieved from the measured Actions artifact by a connected preservation job, independently checked, committed on this same session branch, then fetched and rechecked here. It contains no frozen parent tensor. `reference.json` records complete immutable parent hashes/recipe and frozen/graft identities. It is a **parent-reference-dependent checkpoint**, not a standalone download.

The large private self-contained checkpoint was exported and loaded during the measured run, but is not distributed because of size and model-license restrictions/gaps. No public pretrained release is claimed. Reconstruct without retraining using `scripts/reconstruct_reference.py`; export privately with `--out`. Full instructions are in `docs/pretrained-study.md` and the runnable Colab notebook.

## G. Held-out benchmarking, controls and registration

`configs/pretrained-study.json` registers thresholds and comparison/prompt/decoding/data rules before final held-out results. Calibration/development use GSM8K TRAIN and WikiText TRAIN/VALID only. Source contents are pinned and content-hashed: GSM8K 7,473 train / 1,319 test (MIT); WikiText mirror/card source and CC-BY-SA-3.0/GFDL attribution recorded. Model-generated GSM solution files are not acquired. Exact selected prompt/text overlap is checked; original-parent pretraining contamination is unknown.

Required comparisons are both original parents, calibrated and uncalibrated children, donor bypass, recipient prefix/suffix bypass, input/output interface zero ablations and independently randomized interfaces. This random-interface control is not budget-matched random-sector training, a disclosed limitation. Synthetic engineering records remain separate. No additional reasoning benchmark is measured.

The CPU pilot is predeclared **4 GSM8K test cases, 16 WikiText TEST windows, greedy 32-token generation, 16 calibration steps/candidate**. Its scores are exploratory, not official leaderboard results. Scoring uses last ####/boxed/numeric occurrence; strict #### accuracy also recorded. Coincidental/intermediate numeric matches are possible, particularly at the short output budget. Language reports native-token likelihood/perplexity and cross-vocabulary bits per UTF-8 byte, not official word-level WikiText perplexity.

The larger registered preset uses 64 cases/windows and 128-token generation. Inheritance requires the **64-sample minimum**, source capability evidence, at least max(10% math accuracy, 50% donor accuracy), <=1.10×recipient language BPB, and positive paired 95% contribution bounds. Complementarity additionally requires these conditions together; improvement requires positive paired comparison against both best-parent measures. Wilson/paired-bootstrap uncertainty is single-seed exploratory; contiguous language windows are correlated.

### Preserved failed first controlled attempt

Run **38292930207**, source **de830174d43b8a4ce9c5172b2256fa8899b0e7ec**, failed on a programming defect: the study requested a donor tokenizer field not exposed by `LoadedParents`. The Mamba original-parent observations completed before failure. They are retained under `reports/pretrained/attempt-38292930207/` with raw rows, registered config, sources and result statuses.

That attempt measured Mamba **1/4 flexible numeric matches (25%; Wilson 95% 4.56–69.94%)**, **strict match 0%**, and language **1.7658307429651134 bits/byte** on the 16 selected WikiText windows. These are partial native-parent observations only, not child results or established mathematical ability. All child/donor/ablation metrics in that failed attempt remain **NOT_RUN / null**. The donor tokenizer field was corrected with a regression test; failure evidence was not erased.

### Completed controlled retry — verified negative result

Corrected run **38293921543**, source **95b2831a7f54ae7755e6892e364d993a779a3193**, completed all 12 execution stages. Observed study time was **1,498.081661872 seconds** (about 24.97 minutes; whole Actions job 25m44s). Public bot JSON bundle checksum, raw-file digests, metric recomputation and original pinned dataset references/text hashes were verified independently here. All records live under `reports/pretrained/run-38293921543/`; the previous failed attempt is preserved separately.

| Variant | GSM8K flexible matches (4 cases) | Strict #### accuracy | WikiText bits/byte (16 windows; lower is better) |
|---|---:|---:|---:|
| Original Mamba recipient | 1/4 (25%) | 0% | 1.765828132006639 |
| Original Qwen donor | 1/4 (25%) | 0% | 2.5806685715288404 |
| Calibrated child | **0/4 (0%)** | 0% | **3.444856383352949** |
| Uncalibrated child | 0/4 | 0% | 5.524332142264799 |
| Donor bypass | 0/4 | 0% | 10.32563161071116 |
| Recipient prefix bypass | 0/4 | 0% | 4.264506914218898 |
| Recipient suffix bypass | 0/4 | 0% | 10.769998042271437 |
| Zero input interface | 0/4 | 0% | 5.209031366423994 |
| Zero output interface | 0/4 | 0% | 3.9425284076474734 |
| Independent random interfaces | 0/4 | 0% | 5.712653009342807 |

Math Wilson 95% intervals: for 1/4, **[0.04558726080970055, 0.6993581574175981]**; for 0/4, **[0, 0.4898908364545973]**. These are tiny-sample exploratory estimates, not convincing mathematical capability scores. Qwen's matching response stopped at an intermediate “Each child requires 5”; flexible extraction matched that number, not a completed correct rationale. Both parent strict-format scores are zero. The 32-token budget and prompt/scoring format severely limit interpretation; no published-model benchmark accuracy comparison is claimed.

The child produced repetitive strings such as `the,, the, the,, the, the...` and commas, not useful answers. Its language BPB was **1.9508446608777879× Mamba's** (derived from raw summaries), and worse than Qwen too. Thus this pilot does not demonstrate retention or improvement; the point estimates miss both the math floor and the 1.10× language-retention target. Formal inheritance/source-capability/contribution/complementarity/improvement criteria are **NOT_RUN**, explicitly blocked by the predeclared **64-sample minimum**, rather than retrospectively lowering that minimum to claim success.

Language ablations worsen likelihood relative to the full calibrated child. Derived paired-window bootstrap control-minus-child BPB differences (95% intervals): donor bypass **6.88077522735821 [4.871655295249589, 9.062780649531115]**; prefix bypass **0.8196505308659493 [0.7047022065474255, 0.9791546984080529]**; suffix bypass **7.3251416589184855 [6.758397980655381, 7.961463480203531]**. Math child-minus-donor-bypass difference is **0 [0,0]**. Full paired calculations are in `derived-pilot-analysis.json`. These correlated-window, single-seed effects show computational dependence/fit, **not inheritance of useful parent-specific capability**. Random interfaces are untrained, not a matched-training-budget random-sector control.

#### Actual development search/calibration

The cut remained Mamba [11,13). Independently calibrated donor candidates were [13,15) and [20,22). Equal-weight math/language development NLL scores were **9.439523602347089** and **8.873143710709705**. The predeclared lowest-development-NLL rule selected **Qwen [20,22)**; no test score selected it. No context-oversize math rows were omitted in this run.

Each candidate trained only the eight new interfaces/norms for **16 steps**, LR **0.0003**, batch 1, seed 42, max context 512, checkpointing true. Real calibration data: **16 human GSM8K TRAIN answers + 8 WikiText TRAIN windows**, with **4 train-derived math development answers + 4 WikiText VALID windows**. Text windows are 192 characters. Selected candidate train token NLL **16.635165920505276 → 8.231364559817624**; development overall NLL **8.653751111677009**, math **8.29272531580042**, language **9.45356210561899**. Selected calibration time **232.02033327599997 seconds**. Candidate [13,15) calibration time **264.48450685299997 seconds**. Loss improvements alone remain fitting evidence.

The selected [20,22) child has the same measured parameter counts as the first structural [13,15) child, but different transplanted tensor identities and newly trained interfaces. Original parents remained unchanged. Its frozen-child SHA is `ba6c0f8331f18c222ed4c2b46016464e709702199bdf83e6236033b1a055867d`. Fresh-process parent-free load again produced **identical logits**. The public reference descriptor is in the run records; actual new selected interface tensors are preserved separately under `examples/pretrained-pilot-reference/` after connected extraction/checking, without frozen weights. Its actual interface file is **9,465,560 bytes**, SHA-256 `5f2337211dd54ea0dd001244735cc3e7c3d2b74468eaea99053ad6a84411d537`.

Both original-parent sector development NLL/identity ablations and scalar activations were actually measured; `sector-inventory.json` contains these real observations and complete tensor shapes. The native stock operations remain intact. This does not make the selected contiguous sector a biologically isolated “math organ.”

## H. Reverse surgery policy

The runner implements resource-bounded opposite-direction **structural-only** checking after primary execution/export/reload validates. It does not train a second larger experiment or add sectors blindly. The completed pilot also measured a finite **Qwen → Mamba → Qwen** reverse structural forward (PASS). Reverse training/benchmarks remain **NOT_RUN** because primary capability evidence was negative/underpowered. A finite reverse forward is not inherited capability.

## I. Scientific integrity and tests

The offline suite is **137 passing** after interface-asset/tokenizer/published-evidence tests; lint, format and dependency checks pass. All original tests remain. New checks cover provenance labels, complete sector execution, activation observations, identity ablations, real-text/ground-truth objective masking, calibration journals, bridge-only integrity, byte-normalized scoring, missing/duplicated/tampered/manually substituted metric summaries and evaluation-only controls. These are mostly offline fixtures, never benchmark performance evidence.

Standard Python 3.10/3.11 CI stays offline/lightweight. Real metadata/weight/controlled jobs are separately opted in. An earlier bot-authored preservation commit required PR workflow approval; the integration could not approve that run (403), while subsequent normal-agent push CI ran successfully. Do not equate an unapproved run with failed scientific code.

`verify_study.py` recomputes metrics from raw observations, checks config/record hashes/sample counts/statuses and optionally compares references/text hashes to registered actual datasets. Unexecuted metrics must be null; missing data cannot become score zero. Original frozen tensors and fresh-process behavior are verified. **A malicious author can forge a complete bundle and recalculate all hashes**; these checks are tamper-evident consistency, not cryptographic proof of origin or honest inference. GPU/Colab and multi-seed validation remain NOT_RUN.

## J. Publication and next milestone

All work stays on `arena/97efdf4a-equylaptatemps`, existing PR #1, without force-push/restart/deleting the original baseline. Appropriate code, reports, metadata, real new-interface checkpoint and runnable notebook are committed/pushed; no parent weights or raw large datasets/secrets are published. Public display title remains HETEROGENEOUS AI MODEL SURGERY.

The exact next research milestone is **registered, adequately sized held-out evidence of retained language AND mathematics with paired donor/recipient/interface contribution controls**, then matched-budget random-sector controls and multiple seeds before claiming inheritance/complementarity. If retention fails, diagnose representation/tokenizer/interface mismatch and candidate sensitivities before larger surgery or reverse capability training. Structural transplantation has been achieved; functional inheritance is still a research question.
