# Campaign 4 — Initial Audit

Date: 2026-10-13. Project: **HETEROGENEOUS AI MODEL SURGERY**.

## Verified starting state

| Item | Reported | Verified actual |
|---|---|---|
| Branch | `arena/97efdf4a-equylaptatemps` | same |
| HEAD | `937f3af` | `937f3af9b723660a4e93e21639ccf91a3b9ff6c4`, local = remote |
| PR | #1 | OPEN, head matches, Python 3.10/3.11 CI passing (run `38379650746`) |
| Working tree | — | clean (a session-restore artifact briefly showed files untracked at the base commit; reconciled by fast-forward to the remote head with zero content differences) |
| Local tests | 223 (Campaign 3) | 223 reproduced in a rebuilt environment; 226 after Campaign 4 tests |

Parents (unchanged pinned revisions): Mamba `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1` (width 768, 24 blocks); Qwen `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` (width 1536, 28 blocks). Child: 217,553,920 parameters, 215,187,712 frozen.

Environment: 2 CPUs, 3.8 GB RAM, no GPU, Hugging Face unreachable locally; real-parent work runs only in opted-in GitHub Actions CPU workflows. GPU experiments: `NOT_RUN_RESOURCE_UNAVAILABLE`.

## Architecture audit (source-verified)

Forward pass (`src/equylaptatemps/model.py:166-189`): host embedding → Mamba prefix [0,11) → `input_seam` → Qwen graft [20,22) → `output_seam` → Mamba suffix [13,24) → final norm → tied host head. Donor position handling: `backbones.py:110-113` derives position IDs from the attention mask and applies the donor config's own rotary embeddings inside `StackSegment`; causal mask at `backbones.py:76-83`. There is **no graph-level skip** around the donor: the Campaign 3 bypass is functional — the trained seams make the donor a near-identity passenger (relative update ≈0.09; bypass ablation within 0.2%; random twin better at both budgets). Campaign 3's nine reported observations were re-verified against the committed, hash-checked run records (`campaigns/campaign_03/screen_38367618192`, `confirm_38371104410`, `confirm2_38375777144`) rather than rerun.

## Bypass audit summary

| Bypass | Location | Severity | Disposition in Campaign 4 |
|---|---|---|---|
| Learned seam→seam identity through a near-identity donor | `model.py:169-183` (functional, not structural) | Critical | Donor-delta pathway (Candidate A) + bottleneck interfaces (Candidate B) |
| Full-width (768/1536) interface capacity lets bridges encode a host-friendly code the donor passes through | `sequence_seams.py` SequenceSeam | High | 128-wide bottleneck seams (893,696 trainable vs 3,189,248) |
| Donor internal residual stream lets blocks approximate identity | inside frozen Qwen blocks | Intrinsic | Cannot remove without modifying donor (forbidden); delta pathway exposes it: identity donor ⇒ zero suffix signal |
| Loss rewards only final predictions, never donor use | training objective | High | Architecture-level forcing + Gate C ablation requirement; no teacher targets introduced |

Full machine-readable audit: `campaign_04_bypass_audit.json`; graph description: `campaign_04_architecture_audit.md`.
