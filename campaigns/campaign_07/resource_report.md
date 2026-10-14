# Campaign 7 Resource Report

| Job | Run ID | Wall time (runner-reported) | Contents |
|---|---|---|---|
| Screen | `38542143640` | 735 s | Parent/benchmark download, native references, 2 runs |
| Replication | `38544932532` | 742 s | Same, seed 123 |

GitHub-hosted CPU runners (2 threads enforced), fp32, gradient checkpointing.
Peak memory bounded by the 12 GiB registry floor check; the 143.7M-param child
plus both parents (130M + 135M) fit comfortably — the small-donor pairing cut
job time ~3× vs the Qwen-1.5B campaigns. No parent weights or checkpoints are
stored in Git; run records live under `screen_38542143640/` and
`replication_38544932532/` (JSON only). Actions artifacts
(`campaign07-screen-recovery`, `campaign07-replication-recovery`) are temporary
(~90-day expiry) and are **not** the permanent record — the SHA-verified PR #1
evidence bundles and the imported records in this directory are.
