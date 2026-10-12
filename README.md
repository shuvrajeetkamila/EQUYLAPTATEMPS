# EQUYLAPTATEMPS

## Stranger-model surgery — not LoRA, distillation, or weight merging

A runnable research implementation of **whole-block transplantation between Mamba and Qwen2**, two different architecture and declared weight-lineage families. A contiguous recipient sector is physically removed and replaced by a complete donor sector, connected by two new dense interfaces. The result is **one causal model**, not two models answering independently.

**Status: tested architecture prototype, not a validated pretrained foundation model.** The offline demo uses randomly initialized, tiny versions of the actual Hugging Face Mamba and Qwen2 implementations. No pretrained math/reasoning transfer, superiority, or “best of both parents” result has been established. The pretrained-parent workflow requires separately acquired weights, suitable hardware, supervision, and benchmarks.

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

### The proposed pretrained pair

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

A recorded local run is in [reports/synthetic-smoke.json](reports/synthetic-smoke.json); interpret it with the [synthetic model card](docs/model-card.md). [Local validation evidence](reports/local-validation.json) records **106 passing offline tests**, lint/format checks, dependency versions, and source hashes. Raw generated weights live in ignored `artifacts/`. The explicitly requested tiny child is also available as a [portable synthetic ZIP](examples/equylaptatemps-synthetic-child-v0.1.0.zip), with [verification instructions](examples/README.md). This small archive is the only intentional checkpoint-in-Git exception; it is not a pretrained capability model.

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

`validate` checks declared identity and recipe structure; `build` additionally checks actual local architectures, sector bounds, vocabularies, and available snapshot evidence. A revision supplied manually is **recorded**, not automatically verified against a remote repository.

Build loads both full parents before discarding unused sectors. Plan RAM/VRAM for those loads **and** activations/attention/logits during calibration, not just the seam parameters. There is no quantized, sharded, or low-memory parent loader. CPU float32 is supported but large-parent runs may be prohibitively slow. GPU and large-pretrained execution have not been validated by the checked-in smoke report.

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
