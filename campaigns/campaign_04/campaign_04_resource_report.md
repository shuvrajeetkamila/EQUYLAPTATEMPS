# Campaign 4 — Resource Report

Local sandbox: 2 CPUs, 3.8 GB RAM, no GPU (GPU work NOT_RUN_RESOURCE_UNAVAILABLE),
21 GB disk. Real-parent jobs ran on opted-in GitHub Actions CPU runners
(ubuntu-22.04, 2 threads, float32), sequential, one experiment at a time.

| Job | Run | Runtime | Outcome |
|---|---|---|---|
| Screen attempt 1 | 38390162060 | ~21 min | All training completed; results failed on a post-training token-accounting type error; preserved |
| Screen attempt 2 | 38393669406 | 1698 s job time | 3 candidates completed |
| Twins | 38397294665 | 1565 s | 3 matched twins completed |
| Replication | 38401090731 | 1101 s | seed-123 pair completed |

Each candidate: 32 AdamW steps, batch 1, 2,160 supervised training tokens
processed per run (trace-accounted). No parent weights were committed;
candidate binaries remain in the named Actions recovery artifacts.
