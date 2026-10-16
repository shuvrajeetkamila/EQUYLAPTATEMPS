# Campaign 4 — Recovery Guide

State lives in Git (branch arena/97efdf4a-equylaptatemps, PR #1). To resume:
read campaigns/campaign_04/campaign_04_results.json and the per-run
campaign_04_results.json registries under screen_38393669406,
twins_38397294665, replication_38401090731. Rerunning
scripts/campaign04_runner.py with the same --registry/--out skips
hash-verified completed experiments and resumes interrupted training from
calibration/state checkpoints (fit_resumable verifies config/data/code/
frozen identity before reuse). Candidate binaries and optimizer states are
in GitHub Actions artifacts campaign04-screen-recovery,
campaign04-twins-recovery, campaign04-replication-recovery (run IDs in
campaign_04_results.json); they are not in Git and parent weights must
never be committed. The failed first screen attempt (38390162060) is
preserved under screen_38390162060 with its failure.json records.
