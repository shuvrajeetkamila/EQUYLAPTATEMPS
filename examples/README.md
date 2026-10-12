# Portable synthetic child — NOT a pretrained capability model

The user requested the child artifact on GitHub. The sandbox could push Git but could not reach GitHub's separate release-upload host. Therefore this **single small ZIP** is an intentional exception to the rule against generated checkpoints in Git. Large parent snapshots, private datasets, and all other checkpoints remain ignored.

[Download the synthetic child ZIP](equylaptatemps-synthetic-child-v0.1.0.zip) using GitHub's file download button. Its [SHA-256 checksum](equylaptatemps-synthetic-child-v0.1.0.zip.sha256) verifies the exact archive bytes.

This is a tiny random-parent Mamba/Qwen2 architecture test: complete native block replacement, frozen parent tensors, and 888 newly trained dense seam parameters. It is **not** the pretrained Mamba/Qwen2.5-Math experiment. No arithmetic/reasoning/code ability or “best of both” result is established; byte-level generated output can be nonsense.

## Verify and load

From an installed checkout (`python -m pip install -e .`):

```bash
cd examples
sha256sum -c equylaptatemps-synthetic-child-v0.1.0.zip.sha256
python -m zipfile -e equylaptatemps-synthetic-child-v0.1.0.zip ../artifacts/downloaded-synthetic
cd ..
equylaptatemps inspect artifacts/downloaded-synthetic/child --verify
```

The ZIP contains a self-contained `child/` directory with safetensors weights, metadata, and the recipient tokenizer, plus its model card, synthetic audit, licenses/notices, and `RUN.md`. Neither parent directory is needed to reload it. An offline test verifies the checksum and loads this exact distributed archive.

Inspect the [model card](../docs/model-card.md) and [audit](../reports/synthetic-smoke.json). For a real-parent experiment, follow the [main README](../README.md) and independent [experiment protocol](../docs/experiment-protocol.md).
