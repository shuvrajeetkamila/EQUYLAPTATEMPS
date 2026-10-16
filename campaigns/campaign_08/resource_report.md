# Campaign 8 Resource Report

GitHub-hosted CPU runners, 2 threads, fp32. Runner-reported wall times:

| Job | Run ID | Runtime | Contents |
|---|---|---|---|
| Frozen screen (failed pre-training) | `38561906471` | 61 s | scale-guard failure, preserved |
| Stage D amended (32 steps ×2) | `38563416962` | 786 s | calibrated pair, seed 42 |
| Stage F (32 steps ×2) | `38565839960` | 772 s | calibrated pair, seed 123 |
| Stage E calibrated (96 steps ×2) | `38565837824` | 2,154 s | budget test |
| Stage E original (96 steps ×2) | `38566012225` | 2,407 s | budget test |

Total compute ≈ 1.7 runner-hours for six new controlled runs (plus two stored
Campaign 7 baselines reused, saving a repeat). No GPU was used or claimed.
Weights/checkpoints are not in Git; Actions recovery artifacts expire (~90
days) and are not the permanent record — the SHA-verified imports in this
directory are.
