# Campaign 8 Integrity Report

- Pairing, sector, protocol, metric, and gates identical to Campaign 7's frozen
  preregistration; only interface initialization varied, per Campaign 8's own
  preregistration frozen at commit `3844b6a` before training.
- Parents fingerprinted unchanged in all five Campaign 8 jobs
  (`parents_unchanged: true`); only the two dense seams (888,768 params)
  trainable; twin construction identical to Campaign 7 (native init, seed 1234).
- Calibration statistics came from 8 unlabeled TRAIN windows per child
  (identical procedure for pretrained and twin); no hidden-activation targets,
  no held-out data anywhere; held-out remains `NOT_RUN_NO_PROMOTION`.
- All evidence bundles SHA-256-verified on import (hashes in
  `campaign_08_results.json`). The frozen method's failed runs are preserved
  (`screen_38561906471/`), as are all Stage D/E/F records.
- Known cosmetic defect (pre-existing, not introduced here): the registry
  state file's `campaign` field reads `2` regardless of campaign (hardcoded
  default in `CampaignRegistry` state writer since Campaign 5); filenames,
  registries, and published results carry the correct campaign number.
