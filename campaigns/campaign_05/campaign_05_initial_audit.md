# Campaign 5 — Initial Audit

Date: 2026-10-13. Project: **HETEROGENEOUS AI MODEL SURGERY**.

Verified: branch `arena/97efdf4a-equylaptatemps` at `c1ba46a8cf78e226b0f42b7e2819ca9dfd05d174`
(local = remote, working tree clean, no unpushed or uncommitted work, no jobs running).
PR #1 OPEN with Campaign 4 head; Python 3.10/3.11 CI passing. Campaigns 1–4 reports,
registries, and hash-verified run records are committed on the remote branch — preservation
is by the existing history; nothing was rewritten. Candidate binaries remain only in named
GitHub Actions artifacts (campaign04-*-recovery etc.); recorded, not falsely described as
permanent. Local tests at audit: 227 passed.

Parents (pinned, unchanged): Mamba `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
(768 wide, 24 blocks); Qwen `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`
(1536 wide, 28 blocks). Campaign 4 baseline verified from committed records: bottleneck-128
+ donor-delta child, donor sector [20,22), macro dev NLL 7.7873 vs random twin 8.0091
(+2.8%, seed 42), replication +1.7% (seed 123), language 1.83x host — Outcome B.

Architecture verified in source: host prefix Mamba [0,11), suffix [13,24) (`child_from_parents`,
model.py forward 166-189); donor-delta = graft output minus graft input before the output
interface (model.py, `donor_delta` flag); bottleneck seams in sequence_seams.py; random twins
via controls.random_qwen_sector_control (supports any contiguous sector span).

Environment: 2 CPUs, 3.8 GB RAM, no GPU (GPU NOT_RUN_RESOURCE_UNAVAILABLE); real-parent jobs
run in opted-in GitHub Actions CPU workflows only.
