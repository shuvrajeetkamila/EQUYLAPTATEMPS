# HETEROGENEOUS AI MODEL SURGERY — Campaign 6 Final Report

**Capacity Control: Is the Mandatory-Use Pathway Itself Too Lossy?**
Date: 2026-10-13 · Branch `arena/97efdf4a-equylaptatemps` · PR #1

## Final decision: Outcome C — final negative for this pairing and interface family at this scale

Campaign 5 left one alternative explanation open: perhaps the forced
(bottleneck + donor-delta) pathway destroys so much host information
that no donor could help. Campaign 6 tested exactly that, using random
sectors only. The pathway is indeed the bottleneck: widening it and
tripling training each help, but neither comes close to the
unconstrained child. With that, every tested explanation has been
controlled for, and the programme reaches its defensible conclusion.

## Results (development macro token-NLL, random sectors throughout)

| Architecture | Budget | Macro | Remediable threshold |
|---|---|---|---|
| Unconstrained full-width (Campaign 3 reference) | 32 steps | 4.9024 | — |
| Unconstrained full-width (Campaign 3 reference) | 96 steps | 4.0137 | — |
| Mandatory bottleneck-128 + delta | 32 steps | 8.0091 | — |
| Mandatory bottleneck-256 + delta (new) | 32 steps | 7.3232 | ≤5.393 — FAIL |
| Mandatory bottleneck-128 + delta (new) | 96 steps | 7.0706 | ≤4.415 — FAIL |

Two runs, one sequential job (`38428594451`, 1735 s), 32/96 AdamW steps,
seed 42, parents frozen and fingerprint-verified, decision rule frozen
in `campaign_06_preregistration.md` before training. Held-out
confirmation: NOT_RUN_NO_PROMOTION (this study could not promote by
design).

## Programme-level conclusion (Campaigns 1–6)

1. The transplant is mechanically sound: intact pretrained Qwen blocks
   execute inside a Mamba child, reconstruct exactly, and never change.
2. With free bridges, the child learns to ignore the donor — random
   blocks beat pretrained ones (Campaign 3).
3. With forced bridges, the donor is used and once showed a small
   two-seed benefit (Campaign 4), but sector width/placement scaling
   revealed that benefit as unstable noise (Campaign 5).
4. The forced pathway itself cannot match unconstrained performance
   even with random sectors (Campaign 6).

**Useful heterogeneous capability inheritance was not achieved for the
Mamba-130M / Qwen2.5-Math-1.5B pairing, across dense, sequence-aware,
bottleneck, and donor-delta interfaces, 1–6-block sectors at three
positions, statistical conditioning, and budgets to 96 steps.** This
does not generalize to other pairings, to objectives rewarding donor
information directly, or to far larger training scales; those would be
new hypotheses requiring new preregistration.

## Verification and publication

230 offline tests pass; whole-repo lint/format clean; final-head
Python 3.10/3.11 CI verified in the PR. Evidence:
`campaigns/campaign_06/` (audit, preregistration, registry, raw run
records, results, failure analysis). Candidate binaries remain in the
named Actions artifact; no parent weights are in Git.

## Plain-English closing summary

Across six campaigns we built a working surgical transplant, proved the
inserted chapters were being ignored, rewired the book so they could
not be ignored, discovered the rewiring itself damages the book too
much, and checked every reasonable variation of chapter length,
placement, and doorway width. The transplant is real; the inheritance
is not there. That negative answer is now supported by matched random
controls at every step — which is precisely the kind of evidence this
project set out to produce, whichever way it landed.
