# HETEROGENEOUS AI MODEL SURGERY — Campaign 5 Final Report

**Donor Placement and Sector-Width Scaling under Mandatory Donor Use**
Date: 2026-10-13 · Branch `arena/97efdf4a-equylaptatemps` · PR #1

## Final decision: Outcome C — structural surgery without useful inheritance

Changing the donor sector's width (1, 2, 4, or 6 Qwen blocks) or position
(early, middle, late) does not produce useful inheritance. The donor's
transformation grows with width, but task performance and any
pretrained-specific advantage do not: benefits against matched random
twins scatter around zero, and the single largest benefit (+4.5%, six
blocks) reversed to −2.1% at an independent seed. Every candidate failed
the usefulness gate, nothing was promoted, and held-out evaluation was
not run.

## 1–5. Starting point, preservation, hypotheses

Verified start at Campaign 4 head `c1ba46a` (clean tree, PR #1 open, CI
green, 227 tests). All Campaigns 1–4 history is preserved untouched on
this branch. Hypothesis: Campaign 4's two-block sector [20,22) was too
narrow or badly placed; competing explanations (instability, harmful
scaling, unsuitable boundary) were preregistered with identical gates
A–E before training (`campaign_05_preregistration.md`).

## 6–11. Architecture, matrix, protocol, resources

Unchanged mandatory-use child (bottleneck-128 + donor-delta, host
partition [11,13), frozen parents, 893,696 trainable interface
parameters in every pair). Single variable: donor sector. Twelve
real-parent runs in four sequential Actions jobs (screen 1591 s; twins;
positions; replication 1320 s), each 32 steps / 2,160 supervised tokens,
seed-matched, CPU float32, parents fingerprint-verified.

## 12–18. Results (development macro token-NLL; benefit vs matched twin)

| Sector | Benefit vs twin | Pretrained macro | Language ×host | Donor update |
|---|---|---|---|---|
| 1 block late [20,21) | +0.5% | 8.0971 | 1.87 | 0.048 |
| 2 blocks late [20,22) (C4) | +2.8% | 7.7873 | 1.83 | 0.076 |
| 4 blocks late [20,24) | −0.7% | 7.8621 | 1.82 | 0.128 |
| 6 blocks late [20,26) | +4.5% → −2.1% (seed 123) | 7.7882 | 1.81 | 0.184 |
| 4 blocks early [2,6) | −1.6% | 7.9135 | 1.83 | 0.211 |
| 4 blocks middle [12,16) | −0.5% | 7.8464 | 1.81 | 0.099 |

Donor removal/neutralization remained catastrophic at every sector
(dependence by construction), so dependence alone distinguishes nothing.
Mathematics is the math half of the primary metric throughout (child
math NLL 7.3–7.7 vs host 2.48); no generated-answer claim is made.

## 19–24. Gates, held-out, failures, integrity, tests

Gate B passed only for 6-block late at seed 42 and reversed on
replication; Gate D failed for all (language 1.81–1.87× vs 1.50 ceiling;
macro 7.79–8.16 vs 5.74 ceiling). Promotion: none. **Held-out:
NOT_RUN_NO_PROMOTION.** No failed or interrupted jobs. Parent
fingerprints unchanged; checkpoint hashes in
`campaign_05_checkpoint_integrity.json`. 229 offline tests pass;
whole-repo lint/format clean; final-head CI verified in the PR.

## 25–28. Publication and recovery risks

All code, registries, raw records, and analyses are committed on this
branch (final hash in the PR). Candidate binaries remain only in the
four named Actions recovery artifacts — temporary storage, recorded as
a recovery risk, not permanent preservation.

## 29. Plain-English interpretation

Campaign 4 proved we could force the book's story through the
transplanted chapters. Campaign 5 tried longer and differently-placed
chapters: the chapters now change the story *more*, but the book does
not get *better*, and blank chapters do just as well in most trials —
the one apparent win flipped sign when retried with a different random
start. The honest conclusion is that, for this Mamba/Qwen pairing, this
interface family, these sector ranges, and these budgets, the
transplant's pretrained knowledge provides no stable useful advantage.

## 30. The next justified scientific step

Not another width sweep. The cumulative evidence (Campaigns 3–5) supports
one remaining untested explanation worth a preregistered study: the
*bottleneck+delta pathway itself* destroys too much host information
(all forced children sit ~2 NLL above the unconstrained child), so any
future attempt should first test whether a mandatory-use design can
preserve unconstrained-child performance with a *random* sector — a
pure architecture-capacity control — before asking pretrained blocks to
add anything. If that control also fails, the defensible final
conclusion is a rigorous negative for this pairing at this scale.

Evidence: `campaigns/campaign_05/` (audit, preregistration, sector
matrix, registries, per-run records, screening/failure analyses,
results, controls, ablations, activations, integrity, resource/test/
recovery reports). No parent weights are in Git.
