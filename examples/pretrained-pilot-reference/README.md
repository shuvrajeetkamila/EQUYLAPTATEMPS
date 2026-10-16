# Actual selected pretrained pilot child — NEGATIVE capability result

HETEROGENEOUS AI MODEL SURGERY. Connected run 38293921543, source 95b2831.

New interfaces ONLY, not standalone; frozen parents are not distributed. Whole Qwen blocks [20,22) replace Mamba [11,13), selected on development NLL. Sixteen interface steps per candidate on actual human GSM8K TRAIN answers and real WikiText TRAIN text. No distillation or parent-weight changes.

Held-out pilot: child math 0/4, WikiText 3.444856383352949 bits/byte, worse than both original parents (Mamba 1.765828132006639; Qwen 2.5806685715288404). Child generations repeated `the` and commas. No useful inheritance established. Formal capability criteria remain NOT_RUN because the sample minimum is 64.

```bash
python scripts/reconstruct_reference.py --reference examples/pretrained-pilot-reference --device cpu
```

Requires exact pinned parent safetensors and common float32; identities, tokenizers and frozen/graft hashes verified. See the dated report/raw observations and docs/pretrained-study.md. Parent and dataset terms/gaps remain; no frozen tensor redistribution and no public standalone model release claimed.
