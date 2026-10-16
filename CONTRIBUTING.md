# Contributing

Install `.[dev]`, then run:

```bash
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
equylaptatemps demo --out artifacts/my-smoke --report artifacts/my-report.json
```

Tests must not download weights/tokenizers or require a GPU. Use tiny, randomly initialized stock configs. Do not check in parent snapshots, private datasets, or generated checkpoints. The explicitly requested small synthetic archive in `examples/` is the sole exception; do not add large checkpoints or treat it as a pretrained model.

A new supported parent requires a reviewed lineage entry, source evidence, explicit license review, an architecture runtime implementation, and tests against native parent execution. Do not add an `allow-cousins`, `trust-remote-code`, or provenance-bypass flag.

Changes must preserve:

- Complete native blocks; no parent weight averaging/interpolation.
- Frozen parent tensors and deterministic parent evaluation mode.
- Dense, unfactorized new seams only; no LoRA or teacher objectives.
- Mandatory production graft execution and causality.
- Prompt/padding loss masking, recipient-only vocabulary IDs, and tokenizer integrity.
- Native positional/state-space behavior, precision boundaries, strict portable loading, and tied weights.
- Disjoint training/validation prompts in search and a separate final evaluation split.

When library versions change, repeat decomposition/native equivalence, causality, checkpointed-gradient, float32/BF16 serialization, and all CLI workflow tests. Safetensors-only loading and `trust_remote_code=False` are intentional.

Publish negative results and ablations. Never relabel a synthetic demo as a pretrained success, infer a “math organ” from a model name, or claim benchmark improvements without reproducible measurements.
