# Campaign 8 Recovery Guide

- Permanent record: this directory. `campaign_08_results.json` and both CSVs
  regenerate byte-identically via `python scripts/publish_campaign08.py` from
  the imported bundles (`screen_38561906471/`, `stageD2_38563416962/`,
  `stageF_38565839960/`, `stageEcal_38565837824/`, `stageEorig_38566012225/`)
  plus Campaign 7's stored Stage A records; bundle SHA-256 values are in
  `campaign_08_results.json`.
- Re-run any stage on a ≥12 GiB RAM machine with HF access:
  `python scripts/campaign08_runner.py --registry campaigns/campaign_08/<registry>.json --out artifacts/<name>`
  (registries: `experiment_registry.json` [frozen, fails by design record],
  `experiment_registry_stageD_amended.json`, `experiment_registry_stageE_cal96.json`,
  `experiment_registry_stageE_orig96.json`, `experiment_registry_stageF.json`).
- Actions recovery artifacts expire (~90 days); nothing depends on them.
  Checkpoints were job-local; trained seams are reproducible from the frozen
  registries + seeds.
