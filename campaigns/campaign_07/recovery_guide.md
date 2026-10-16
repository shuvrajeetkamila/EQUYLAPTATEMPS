# Campaign 7 Recovery Guide

- **Permanent record:** this directory. `campaign_07_results.json` is derived
  from `screen_38542143640/` and `replication_38544932532/` by
  `scripts/publish_campaign07.py`; every imported file came from a PR #1
  evidence bundle whose SHA-256 is recorded in `campaign_07_results.json`
  (`evidence` section) and in `integrity_report.md`.
- **Re-derive:** `python scripts/publish_campaign07.py` regenerates the results
  JSON and both CSVs byte-identically from the imported records.
- **Re-run from scratch:** on a ≥12 GiB RAM machine with HF access,
  `python scripts/campaign07_runner.py --registry campaigns/campaign_07/experiment_registry.json --out artifacts/campaign07-screen`
  (and the `_replication` registry). The runner re-downloads the pinned
  parents, re-verifies fingerprints, and refuses revision drift.
- **Temporary artifacts:** `campaign07-screen-recovery` /
  `campaign07-replication-recovery` Actions artifacts expire (~90 days) and may
  already be gone; nothing depends on them.
- **Checkpoints:** per-run calibration checkpoints were job-local and not
  preserved (consistent with Campaigns 3–6); the trained seam tensors are
  reproducible from the frozen registry + seed.
