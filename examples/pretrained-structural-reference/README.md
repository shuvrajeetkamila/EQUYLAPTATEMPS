# Actual pretrained structural child: bridge-only checkpoint

HETEROGENEOUS AI MODEL SURGERY. Measured on GitHub run 38290378364, source f812e6b.

This contains ONLY the eight newly trained full-sized interface/norm tensors. The real Mamba and Qwen frozen tensors are NOT bundled. This is NOT standalone.

217,553,920 integrated parameters: 215,187,712 frozen and 2,366,208 new trainable. Whole Qwen blocks [13,15) replace Mamba blocks [11,13). Eight steps on four short WikiText TRAIN windows; no distillation.

Forward, checkpoint export, and exact fresh-process reload were measured. A generation sample was `io`; no useful capability or inheritance established. Do not treat fitting loss as a benchmark score.

Reconstruct on a connected >=12 GiB RAM machine with the repository installed:

```bash
python scripts/reconstruct_reference.py --reference examples/pretrained-structural-reference --device cpu
```

Requires ~3.6 GB pinned public parent safetensors, common float32 loading, and the exact recipe/provenance in reference.json. Frozen/graft and tokenizer hashes are verified. Use --out for a large private standalone child.

Code is Apache-2.0. Dataset terms remain: WikiText CC-BY-SA-3.0/GFDL; see the pinned source/card in the dated research report. Qwen is Apache-2.0. Mamba's official original checkpoint declares Apache-2.0, but the HF conversion omits a direct license tag/document. No frozen weights are redistributed here.
