# HETEROGENEOUS AI MODEL SURGERY

Repository and Python/CLI names (`EQUYLAPTATEMPS`, `equylaptatemps`) are retained solely for compatibility with the existing work. The project is **HETEROGENEOUS AI MODEL SURGERY**.

## Stranger-model surgery — not LoRA, distillation, or weight merging

A runnable research implementation of **whole-block transplantation between Mamba and Qwen2**, two different architecture and declared weight-lineage families. A contiguous recipient sector is physically removed and replaced by a complete donor sector, connected by two new dense interfaces. The result is **one causal model**, not two models answering independently.

**Status: a genuine pretrained heterogeneous child has run and reloaded; meaningful capability inheritance remains UNPROVEN.** On 2026-10-12, connected CPU execution loaded the actual pinned Mamba-130m-hf and Qwen2.5-Math-1.5B weights, transplanted whole blocks, calibrated only new interfaces on real WikiText training text, and reproduced bit-identical logits in a fresh process. No useful math/language inheritance or superiority is claimed. The original random-parent demo is preserved for engineering comparisons only.

### Real pretrained progress (2026-10-12)

- **217,553,920** integrated parameters: **215,187,712 frozen**, **2,366,208 new full-sized interface/norm parameters**.
- Mamba prefix [0,11) → dense 768→1536 interface → **entire Qwen blocks [13,15)** → dense 1536→768 interface → Mamba suffix [13,24)/norm/head. Original Mamba [11,13) removed.
- Eight real WikiText TRAIN calibration steps: diagnostic token NLL **13.13195 → 7.02681**; this is fitting, **not a benchmark or inheritance score**.
- Both original-parent tensor hashes unchanged; private self-contained export and fresh-process inference verified exactly. Short generation was `io`: useful output has not been established.
- **Actual new interface weights**, not a template: [bridge-only child checkpoint](examples/pretrained-structural-reference/README.md). Each reference contains only 9.5 MB of new tensors; the [selected negative-pilot child](examples/pretrained-pilot-reference/README.md) is also preserved. Frozen parent weights are not bundled.
- [Verified execution JSON](reports/pretrained/2026-10-12-cpu-structural.json), [connected run](https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS/actions/runs/38290378364), [real study protocol/instructions](docs/pretrained-study.md), [runnable Colab notebook](notebooks/pretrained-study-colab.ipynb), and [registered configuration](configs/pretrained-study.json).
- **Completed real held-out CPU pilot:** calibrated child **0/4 math**, **3.445 WikiText bits/byte**, worse than Mamba **1.766** and Qwen **2.581** (lower is better). Both original parents had 1/4 flexible numeric matches, strict-format 0%; tiny sample/32-token budget cannot establish their math capabilities. Child output was repetitive. Ablations show computational dependence, **not useful inheritance**. See the [dated negative-result report](reports/2026-10-12-pretrained-surgery.md) and [raw run records](reports/pretrained/run-38293921543/results.json). Formal criteria remain NOT_RUN below the preregistered 64-sample minimum.

**Licensing caveat:** Qwen's pinned snapshot has Apache-2.0. The HF-converted Mamba repository lacks a direct license document/tag; the official original Mamba checkpoint declares Apache-2.0. This is lineage evidence, not a license document in the conversion. Frozen weights are deliberately not redistributed. WikiText retains CC-BY-SA-3.0/GFDL terms; GSM8K is MIT.

### Audited follow-up: matched donor control (2026-10-13)

The repository was independently re-audited at `20f456b`, not the historical `b077573`. New execution-contract and inheritance-verdict guards were committed before further real-weight work.

A connected **64-fresh-window language diagnosis** restored the actual selected child and trained a frozen random donor-sector control with exactly the original interface budget. Results (bits/byte, lower is better): **Mamba 1.792**, **Qwen 2.343**, **real learned-sector child 3.280**, **matched random-sector control 3.044**. Registered language retention and learned-sector advantage both **FAIL**. Ablation dependence did not turn into useful inherited ability. Mathematics was not rerun in this language-only diagnosis; the prior actual math pilot remains separately preserved.

The actual reference-dependent child also passed **exact fresh-process logits AND generation repeatability** with new operator/config contracts; all frozen parent/control tensors remained unchanged. [Follow-up report](reports/2026-10-13-stranger-surgery-followup.md), [raw records](reports/pretrained/run-38301210044/results.json), and [audit](reports/audits/2026-10-13-initial-audit.md). No larger/reverse capability experiment launched blindly; no random sector is called a pretrained transplant.

### Completion-aware math work (2026-10-13)

A separately registered longer-output GSM8K diagnostic uses **16 fresh questions**, two human TRAIN few-shot examples, up to **256 new tokens**, and completed final-number scoring—not arbitrary intermediate-number matches. [Registration](configs/pretrained-completed-math-validated.json), [decoder/limitations](docs/incremental-decoding.md).

Native request-local Qwen KV and Mamba convolution/SSM caches accelerate the same child. Original weights, block indices/configs, one-path topology and default full-prefix training/inference remain unchanged. Supported cache scope: **CPUfloat32, batch1, no padding**. Real cached-versus-original numerical/token checks gate every benchmark; passing unit fixtures alone do not establish real correctness/capability.

The first real-weight attempt **38307630994** failed the **zero-input-interface** numerical gate before scoring test answers. It is preserved in `reports/pretrained/attempt-38307630994/`. That cached control is explicitly blocked; the original uncached control remains. No tolerance was loosened. The registered restricted retry is separate; results are published only after raw-source verification. None of this implies inherited math skill or reverses the negative language results.

### Campaign 2: representation-alignment audit and negative confirmation (2026-10-13)

The original real-parent negative baseline was rerun without retraining the published child. On 64 fresh WikiText windows, Mamba scored **1.792**, Qwen **2.343**, the learned-sector child **3.280**, and a same-budget frozen random-sector control **3.044** bits/byte. A completed-answer diagnostic reproduced **Qwen 12/16 versus child 0/16**. Fresh reconstruction reproduced logits and generation; all original tensors remained unchanged. The full report is [Campaign 2](HETEROGENEOUS_AI_MODEL_SURGERY_CAMPAIGN_02.md).

Measured native Qwen sector input RMS was about **29.6**, while the hybrid supplied about **0.82** on the same unlabeled TRAIN text. Source-covariance/RMS conditioning improved the primary topology’s development loss, but failed the preregistered promotion gate; no candidate was promoted to a new held-out claim. One-block, early-boundary, and reverse structures were cheaply screened and remained poor before calibration. **Outcome C: structural surgery without useful inheritance.** Larger sectors, two-layer bridges, second-seed confirmation, general reasoning, and GPU/Colab remain **NOT_RUN**, rather than being launched blindly after the gate failed.

Machine-readable evidence: `campaigns/campaign_02/campaign_02_results.json`, baseline replay `campaigns/campaign_02/baselines/replay_38354043771/`, and development matrix `campaigns/campaign_02/matrix_38358962201/`. Candidate binaries are retained in the identified Actions recovery artifact; no frozen parent weights were added to Git.

### What is actually implemented

- Intact native Mamba state-space blocks and Qwen2 attention/MLP blocks; either transplantation direction works.
- Complete contiguous block replacement, including all parameters within each selected block.
- Different hidden widths, depths, native positional machinery, and vocabularies.
- A reviewed parent registry that rejects same-family/cousin pairs and overlapping declared weight lineages.
- Two newly initialized, **full-sized dense seam matrices** plus their normalizers. Only these new parameters train; there is no low-rank factorization or parent-weight update.
- Ground-truth response-token supervision, with gradients through the frozen donor and recipient suffix. No teacher model, parent-generated targets, or parent-logit loss.
- Held-out donor-sector search, native-parent baselines, graft ablations, per-domain loss, and optional strict answer matching.
- Portable child-only safetensors checkpoints, checksums, frozen-tensor audits, tokenizer fingerprints, and preservation of available parent license notices.
- Offline tests, a CPU smoke test, and GitHub Actions.

### The actual pretrained pair (capability roles still hypotheses)

| Role | Parent | Architecture | Intended contribution — not a measured result |
|---|---|---|---|
| Recipient | `state-spaces/mamba-130m-hf` | Mamba selective state-space | Recipient embedding, sequence-processing prefix/suffix, output vocabulary/head |
| Donor | `Qwen/Qwen2.5-Math-1.5B` | Qwen2 causal transformer | A candidate decoder sector from a math-oriented checkpoint |

These are not two Qwen fine-tunes or two Mamba derivatives. “Stranger” here means **different supported architectures and disjoint documented/declared weight lineages**, with distinct tokenizer contracts checked at local build time. It cannot prove secret ancestry or absence of overlapping training data. Download records are locally checked, not signed attestations.

The default donor window is a **hypothesis**, not an identified “math organ.” Capability is distributed across blocks, embeddings, and training. Use sector search and a separate final test set; a surgical graft can make a model worse.

## Architecture

```text
recipient token IDs
        |
recipient embedding + native prefix blocks       frozen
        |
new LayerNorm + dense recipient -> donor seam     trainable
        |
ENTIRE native donor block sector                 frozen, mandatory
        |
new LayerNorm + dense donor -> recipient seam     trainable
        |
recipient native suffix + norm + output head     frozen
        |
recipient-vocabulary next-token logits
```

The removed recipient blocks and unused donor embedding/head are **not in the exported child**. Donor token IDs are never fed to recipient embeddings. The donor sector processes projected hidden states on the **recipient token timeline**, using its own native attention or state-space computation. Learning a useful translation between these representations is an unresolved empirical problem, not something a width projection guarantees.

Production inference has no graft bypass. Explicit bypass/zero ablations are evaluation-only. Inserting transformer blocks means the child does **not** retain Mamba's pure state-space complexity or unlimited-context properties.

See [architecture and limitations](docs/architecture.md) for the equations and runtime contract.

## 1. Run the offline demo

Python 3.10+; tested locally with Python 3.11, PyTorch 2.5.1, Transformers 4.48.3. No GPU or model download is needed for this demo.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  equylaptatemps demo --out artifacts/demo --report artifacts/demo-report.json

pytest -q
ruff check .
ruff format --check .
equylaptatemps inspect artifacts/demo --verify
```

CPU-only machines can install `torch==2.5.1` from the official PyTorch CPU wheel index before installing this package, instead of downloading CUDA dependencies. That index is not accessible from every sandbox.

The demo creates a portable synthetic child, calibrates its seams for 40 steps, verifies all frozen tensors stayed unchanged, runs graft ablations, and verifies bit-identical logits after saving/loading. Its random byte-level output is **not evidence of math ability**.

A recorded local run is in [reports/synthetic-smoke.json](reports/synthetic-smoke.json); interpret it with the [synthetic model card](docs/model-card.md). [Local validation evidence](reports/local-validation.json) records **106 passing offline tests**, lint/format checks, dependency versions, and source hashes. Raw generated weights live in ignored `artifacts/`. The explicitly requested tiny child is also available as a [portable synthetic ZIP](examples/equylaptatemps-synthetic-child-v0.1.0.zip), with [verification instructions](examples/README.md). The synthetic archive remains unchanged. The other explicit small checkpoint exception contains only newly trained real-child interfaces, never frozen parent weights.

## 2. Prepare real parents on a machine with Hub access

This repository's original development sandbox could access GitHub/PyPI but **not Hugging Face**, and had no GPU. Pretrained parents were not downloaded or trained there.

On an appropriately connected machine:

```bash
python scripts/prepare_parents.py \
  --models models \
  --out artifacts/local-parents.json
```

This separate acquisition script resolves exact parent commits, downloads safetensors (never pickle weights), checks available Hub LFS SHA-256 hashes, writes a local snapshot inventory, and produces a pinned recipe. It does **not** create or validate the child. It does not request or store credentials.

Alternatively, supply two complete local model/tokenizer snapshots and fill in [configs/mamba-qwen-math.template.json](configs/mamba-qwen-math.template.json). The template intentionally fails validation until actual 40-hex revisions replace its placeholders. Relative snapshot paths are resolved against the recipe's directory.

Inspect **each parent's license** before using or redistributing its weights. The software's Apache-2.0 license does not supersede model-weight licenses.

## 3. Build and calibrate

```bash
equylaptatemps validate artifacts/local-parents.json

# Prefer bfloat16 on compatible hardware; choose a NEW output directory each time.
equylaptatemps build \
  --recipe artifacts/local-parents.json \
  --out artifacts/child-initial \
  --dtype bfloat16 --device cuda

equylaptatemps train \
  --child artifacts/child-initial \
  --data data/local/train.jsonl \
  --validation-data data/local/validation.jsonl \
  --out artifacts/child-calibrated \
  --steps 2000 --batch-size 2 --max-length 256 --lr 0.0003 \
  --gradient-checkpointing --device cuda
```

`validate` checks declared identity and recipe structure; `build` also requires verified pinned snapshot inventories for pretrained IDs, rejects byte-tokenizer fixtures, and checks actual architectures, sector bounds and vocabularies. A manually declared revision alone is insufficient. Inventories/checksums are not signed origin attestations.

Build loads both full parents before discarding unused sectors. Plan RAM/VRAM for those loads **and** activations/attention/logits during calibration, not just the seam parameters. There is no quantized, sharded, or low-memory parent loader. CPU float32 is supported but large-parent runs may be prohibitively slow. Large-pretrained CPU structural execution is now measured in the dated real-parent report; GPU execution remains unvalidated. The older synthetic smoke report proves only engineering behavior.

Ground-truth JSONL format:

```json
{"prompt":"Compute 13 * 7. Return only the number.","answer":"91","domain":"math"}
```

`domain` is optional. Prompt/padding tokens are excluded from the loss. Answers are never silently truncated. The tiny hand-authored files in `data/*.example.jsonl` demonstrate formatting and split hygiene; they are **not sufficient training data or credible capability benchmarks**.

The inference prefix is `prompt + "\n\nAnswer:"`; the answer is tokenized separately with a leading space and an EOS. This is a plain completion protocol, not automatic chat-template support.

## 4. Select sectors rather than assuming which layers are best

```bash
equylaptatemps search \
  --recipe artifacts/local-parents.json \
  --data data/local/train.jsonl \
  --validation-data data/local/validation.jsonl \
  --donor-depth 8 --stride 4 \
  --steps 500 --batch-size 2 --max-length 256 \
  --gradient-checkpointing --dtype bfloat16 --device cuda \
  --out artifacts/selected-child
```

Each candidate uses intact donor blocks and independently initialized seams. Search holds the recipient cut fixed and ranks complete donor windows by held-out response-token NLL. Only the winner is exported. **Do not use the final test set for selection.** Validation loss is not a claim that a sector exclusively contains math or reasoning.

## 5. Compare, ablate, and generate

```bash
equylaptatemps evaluate \
  --child artifacts/child-calibrated \
  --data data/local/test.jsonl \
  --ablations --generate-answers --max-new-tokens 64 \
  --recipient-baseline /absolute/path/to/pinned/mamba-snapshot \
  --donor-baseline /absolute/path/to/pinned/qwen-snapshot \
  --device cuda --report artifacts/final-evaluation.json

equylaptatemps generate \
  --child artifacts/child-calibrated \
  --prompt 'Compute 13 * 7. Return only the number.' \
  --max-new-tokens 32 --device cuda
```

Baseline paths must match recorded families and tokenizer fingerprints. Baseline checkpoints are user-supplied and are not automatically authenticated by this command. Compare child/recipient NLL in the same vocabulary; **do not compare raw donor-token perplexity across different tokenizers**. Strict string equality is only a smoke metric. Real math, reasoning, language-retention, and code correctness need appropriate task scoring and error analysis.

Generation deliberately recomputes the full prefix: no mixed KV/SSM caching is implemented. Runtime is pinned to eager Qwen2 attention and stock Mamba; only nonempty, right-padded batches are supported. Sliding-window attention and arbitrary remote architectures are rejected.

See the [experiment protocol](docs/experiment-protocol.md) before making transfer claims.

## Checkpoints and GitHub contents

A checkpoint contains `child.json`, `model.safetensors`, the recipient tokenizer, and available parent notices. It can reload without either parent directory. There is no call to `torch.load`, no model-generated supervision, and no parent-weight interpolation in the surgery code.

Git contains the implementation, recipes, small hand-authored data examples, documentation, tests, CI, synthetic audit reports, and one explicitly requested small synthetic checkpoint archive. Local parent snapshots, private datasets, virtual environments, and all other generated checkpoints are deliberately ignored. Recreate the tiny child with `demo`; put any permitted large trained artifact in an appropriate model store rather than ordinary Git.

**No pretrained child checkpoint, benchmark superiority, or guaranteed capability inheritance is being claimed.**
