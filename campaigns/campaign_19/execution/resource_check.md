# Resource check (Stage 1, from actual runs)

GitHub Actions ubuntu-22.04 CPU (no GPU), Python 3.11, torch
2.5.1+cpu, transformers 4.48.3. Gates job: ~4 min wall including
downloads. Training job (10 runs): ~2 h wall. Parents fp32 in CPU
RAM; per-run calibration journals preserved in the bundles.
Checkpoints stored: connector calibration state per run
(`train_38950005531/runs/*/calibration/`); no parent weights
committed.
