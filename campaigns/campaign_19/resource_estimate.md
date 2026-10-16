# Campaign 19 — Resource estimate (from Campaign 14–18 actuals)

- Downloads: SmolLM-135M (~270 MB) + Mamba-130M (~260 MB) from pinned
  revisions; benchmark mirrors as in prior campaigns.
- Jobs (GitHub Actions CPU, ubuntu-22.04): diagnose/gates ≈ 10 min
  (Campaign 17/18 pattern); training 10 runs × 96 steps ≈ 40–70 min
  (Campaign 15/16 panel pattern); held-out ≈ 10 min, only on success.
- Memory: two 130M-class parents + child in fp32 — within the
  7 GB runner envelope used since Campaign 14 (peak RSS recorded in
  prior manifests ≈ 2–3 GB). Checkpoints: connectors only (~7 MB).
- No GPU required; no large artifacts committed (bundle mechanism).
