# HETEROGENEOUS AI MODEL SURGERY — real pretrained study

## What is already measured

On **2026-10-12**, connected CPU run
[38290378364](https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS/actions/runs/38290378364)
loaded actual immutable Mamba and Qwen checkpoint tensors, not random fixtures.
See `reports/pretrained/2026-10-12-cpu-structural.json` and the dated research report.

The integrated graph has 11 native Mamba blocks → new LayerNorm/full dense 768→1536
interface → **whole native Qwen blocks [13,15)** → new LayerNorm/full dense
1536→768 interface → 11 native Mamba blocks → original Mamba final norm/head.
Mamba blocks [11,13) are absent. The donor embedding/head are absent. Layer ranges
are zero-based half-open intervals. Every retained block was called exactly once
in the measured default forward trace.

- 217,553,920 unique parameters; 215,187,712 frozen; **2,366,208 new trainable**.
- Common float32, no quantization. Qwen's stored BF16 values are promoted exactly;
  float32 execution differs from published BF16 runs, so parent baselines use float32 too.
- Eight steps on four real WikiText TRAIN windows, each 64 characters.
- Diagnostic train token NLL 13.131952429330477 → 7.0268093642368115.
- Four WikiText validation windows: diagnostic token NLL 8.35311758858817.
- Both complete original-parent hashes and retained frozen/graft hashes unchanged.
- Self-contained private export and fresh-process load with parent loading forbidden:
  **identical finite logits**, shape `[1,4,50280]`.
- A short generated sample was `io`. Language usability/meaningful inheritance was NOT established.

The actual **9,465,560-byte** newly trained interface checkpoint is published in
`examples/pretrained-structural-reference/`. It contains only eight new interface/norm
parameter tensors. It is **not standalone**: the original pinned parents must be acquired.
No frozen parent tensors are uploaded. The earlier tiny synthetic ZIP remains unchanged.

## Completed controlled pilot: negative capability evidence

Connected run **38293921543** completed all parent baselines, candidate calibration,
child/control held-out records, fresh-process reload, frozen-parent audit, and reverse
structural forward. Selected Qwen sector [20,22) had the lowest development loss;
16 steps/candidate used human GSM8K TRAIN references and WikiText TRAIN text only.
Child math was **0/4**, language **3.444856383352949 BPB**, versus Mamba **1.765828132006639**
and Qwen **2.5806685715288404**. Both parents had 1/4 flexible numeric matches and strict
format 0%; incomplete/coincidental numeric matches mean those are not compelling math evidence.
Child generations repeated `the` and commas. The ~25-minute CPU pilot is negative,
underpowered evidence, NOT successful inheritance. Ablations worsened language likelihood
but do not establish useful transferred skills. Formal criteria remain NOT_RUN at min64.
See `reports/pretrained/run-38293921543/` and the dated report for exact prompts/records,
confidence intervals, candidate losses, all controls, failures and limitations.

## Parents, identity and licensing

| Role | Exact model | Immutable revision | Architecture |
|---|---|---|---|
| Recipient | `state-spaces/mamba-130m-hf` | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | selective state-space/causal convolution; width 768, 24 blocks |
| Donor | `Qwen/Qwen2.5-Math-1.5B` | `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` | GQA causal transformer/RoPE/MLP; width 1536, 28 blocks |

Observed parent parameter counts are 129,135,360 and 1,543,714,304. Their model
configurations, tokenizer files, full weight-file digests and notices are recorded
in the reports. They are materially heterogeneous computational families, not two
sizes/fine-tunes of one family. The registry lists disjoint declared weight roots;
this cannot prove absence of undisclosed ancestry or shared training corpora.

Qwen's exact snapshot has an Apache-2.0 LICENSE and license tag. **Mamba's HF-converted
repository omits a direct license tag/document.** The original official checkpoint
`state-spaces/mamba-130m` at `c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0` declares
Apache-2.0. That is lineage licensing evidence, not a claim that the converted repository
contains a license. Redistribution of frozen weights remains deliberately avoided.
Project code licensing does not override model or dataset rights.

Only safetensors is acquired/loaded; no pickle conversion, remote code, LoRA,
weight arithmetic, teachers, separate-model ensemble, or answer/task routing is used.
The default child is one integrated unconditional tensor path. Explicit **evaluation-only**
ablation switches are controls, not the claimed child or a task router.

## Sector inventory and selection

`sectors.py` inventories whole block boundaries, tensor names/shapes/dtypes, parameter
counts, frozen hashes, hidden widths, dependencies, and missing observations explicitly.
The controlled runner additionally measures input/output means/RMS, residual changes,
nonfinite counts and original-parent identity-sector ablation development NLL.
It keeps aggregate statistics only; **parent activations never become training targets**.

The registered study fixes the Mamba cut [11,13), searches Qwen windows [8,10), [13,15),
[20,22), and independently calibrates new interfaces with the same seed/budget.
The CPU pilot uses the last two windows. Selection is the lowest mean of math and
language **development** token NLL in the same recipient vocabulary. This is a limited
search, not proof of an isolated “math organ,” a globally best sector, or functional localization.
Identity hooks remove a parent sector's effect but still compute that sector;
performance ablation is not a throughput claim.

Mamba state/causal convolutions use the stock full-sequence fallback without persistent
cross-request state. Qwen has eager causal masks, native norms/MLP/GQA and config-derived
FP32 RoPE. Native residual and norm operations are retained. No KV/SSM child cache has
been validated; greedy generation recomputes full prefixes and can be slow.

## Actual legally sourced data; no test-answer training

`benchdata.py` fetches immutable GitHub Git blobs, verifies Git object identities and
registered SHA-256 content hashes, and stores raw files in ignored local storage.

- GSM8K: `openai/grade-school-math` at `3101c7d5072418e28b9008a6636bde82a006892c`,
  **7,473 train / 1,319 test** rows; original MIT license copied/hashed.
  `example_model_solutions.jsonl` is NOT acquired or used.
- WikiText-2: PyTorch mirror at `acc295dc7b90714f1bf47f06004fc19a7fe235c4`.
  Data terms are **CC-BY-SA-3.0 / GFDL**, NOT the mirror's software license.
  Pinned dataset-card evidence: `huggingface/datasets` at
  `d7dfbc83d68e87ba002c5eb2555f7a932e59038a`, `datasets/wikitext/README.md`.
  Original dataset attribution: Salesforce WikiText, English Wikipedia authors.
  The mirror is word-preprocessed text (including whitespace/`@-@` conventions).

Calibration mixes explicit HUMAN GSM8K train answers and real WikiText TRAIN next tokens.
Math development is a seeded split of GSM8K TRAIN; language development uses WikiText VALID.
Benchmark TEST targets are read only for scoring. Oversized math calibration records
are filtered with IDs logged, never silently truncated. Exactly normalized selected
question/text overlaps are rejected across calibration/development/test. This does not
perform semantic decontamination or detect original-parent pretraining exposure.

## Registered benchmarks and controls

`configs/pretrained-study.json` is sealed/checksummed before held-out observations.

- Both original parents, calibrated child, uncalibrated child.
- Donor bypass; recipient prefix/suffix bypass; zero input/output interfaces.
- Independently initialized random-interface child, same topology and frozen sectors.
  This is **not** a training-budget-matched random-sector experiment; it cannot alone
  rule out generic nonlinear feature fitting. That remains a follow-up control.
- Existing synthetic child is engineering-only; its scores never count toward inheritance.
- Additional established reasoning benchmark: NOT RUN, explicitly outside this first scope.

Math: identical zero-shot plain base-model prompt, greedy decoding, numeric equality;
last `####` number, then `boxed`, then last numeric occurrence; strict `####` accuracy
also reported. Flexible matching can accept a coincidental/intermediate number.
Exact decoding/sample counts/prompt templates/raw generated IDs and text are recorded.
Language: real WikiText TEST nonoverlapping character windows, BOS context and EOS target;
native token NLL/perplexity and bits per UTF-8 byte. **Native perplexities across vocabularies
are not directly comparable; this is NOT official word-level WikiText perplexity.**

The exploratory CPU pilot predefines 4 GSM8K test cases, 16 WikiText windows, 32 generated
tokens, 16 calibration steps per candidate. Short generation may stop before reasoning
finishes. It cannot pass the registered **64-sample minimum** for inheritance verdicts.
The larger Colab preset uses 64 cases/windows and 128 generated tokens; results from
these protocols are not interchangeable or published leaderboard scores.

GSM8K uncertainty uses Wilson 95% intervals; contribution uses seeded paired bootstrap
intervals. One seed and correlated WikiText windows limit inference. The child must
retain at least max(10% absolute math accuracy, half the demonstrated donor accuracy),
and language BPB at most 1.10×recipient. Source math must meet 10%; source language must
be <=3 BPB. Functional evidence requires positive paired 95% bounds for math versus
donor bypass and language versus both recipient-block bypasses. Complementarity requires
source capability, retention and functional evidence together. Improvement additionally
requires positive paired bounds against the best parents on both primary measures.
A forward pass, fitting loss or any ablation change alone is never inheritance evidence.

Reverse surgery is resource-bounded **structural-only**, after primary export/reload succeeds.
No reverse capability claim or larger sector scaling without positive primary evidence.

## Exact connected-machine / Colab commands

Choose a GPU runtime (T4/16 GB or larger), >=12 GiB system RAM and free disk. The sandbox
has no GPU, ~3.85 GiB RAM and blocked Hugging Face access; do not run full acquisition there.
The observed connected CPU runner had ~15.6 GiB RAM and could run the structural experiment.
GPU/Colab execution itself has NOT been measured in this session.

```bash
git clone --branch arena/97efdf4a-equylaptatemps https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git
cd EQUYLAPTATEMPS
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[dev]'
python scripts/run_pretrained_study.py --config configs/pretrained-study.json --out artifacts/real-study-001
```

Or on a connected >=12 GiB CPU machine, run the explicitly underpowered preset:

```bash
python scripts/run_pretrained_study.py --cpu-pilot --out artifacts/cpu-study-001
```

Outputs are incrementally written, preserving failed/partial records. Always choose a NEW
output directory. Parent/data caches are reused and reverified. Ordinary CI stays offline
and small. Opt-in Actions commit markers are `[parent-probe]`, `[pretrained-cpu-probe]`,
`[pretrained-pilot]`. See `notebooks/pretrained-study-colab.ipynb` for the executable workflow.

Reconstruct either actual child without retraining (the 8-step structural child or selected negative-pilot reference):

```bash
python scripts/reconstruct_reference.py --reference examples/pretrained-structural-reference --device cuda
python scripts/reconstruct_reference.py --reference examples/pretrained-pilot-reference --device cuda
# Optional private standalone export (contains frozen parent-derived weights):
python scripts/reconstruct_reference.py --reference examples/pretrained-structural-reference --device cuda --out artifacts/private-restored-child
```

## Integrity, evidence and limitations

`results.json` separates PASS / FAIL / BLOCKED / NOT_RUN; unexecuted metrics are null,
not zero. Observations/step journals include timings; summary scores are recomputed from
raw records. Registered-config hashes, observation hashes, sample counts, duplicate IDs,
metric consistency and unsafe paths are checked. Missing/substituted/tampered summary
records fail validation. Original frozen tensors are rehashed after calibration/evaluation.
Fresh-process export/reload is checked independently with parent loading forbidden.

**These checks cannot authenticate a maliciously forged entire bundle.** Public Actions
source commits/comments and raw evidence strengthen traceability but are not cryptographic
proof of honest execution or undisclosed ancestry. Baseline/experimental failures remain
in versioned reports; no failed run is erased to make a success claim.

Current physical transplantation success does not establish inherited math/language
capabilities, useful complementarity, deployment readiness, or superiority. Token/hidden
semantic alignment, calibration scale, full matched controls, cache correctness/throughput,
pretraining contamination and multi-seed retention still require research.
