# HETEROGENEOUS AI MODEL SURGERY — Campaign 4 Final Report

**Mandatory Donor-Use Architecture and Pretrained-versus-Random Controls**
Date: 2026-10-13 · Branch `arena/97efdf4a-equylaptatemps` · PR #1

## Final decision: Outcome B — partial evidence of donor contribution, without useful inheritance

Campaign 4 forced the child to actually use its transplanted Qwen sector.
The forcing worked, and for the first time the pretrained donor showed a
small, reproducible advantage over a matched random-sector twin. But the
forced children perform far too poorly to count as useful inheritance, so
no candidate was promoted and no held-out evaluation was run.

## 1. Executive summary

Three mandatory-use candidates were screened against matched random twins
on development data. The bottleneck+delta design (128-wide interfaces; the
suffix receives only the donor's *change* to its input) both depended on
its donor (removal: +49% loss) and beat its random twin at two seeds
(+2.8%, +1.7%). Every candidate failed the useful-performance gate
(language 1.82–1.85× host vs the 1.50 ceiling). The donor is now being
read; what it contributes is not yet worth much.

## 2–5. Starting point and what Campaign 3 established

Verified start: head `937f3af`, PR #1 open, CI passing, 223 tests green.
Campaign 3 (Outcome C) showed sequence-aware bridges improved the child
but the donor was a near-identity passenger (relative update ≈0.09;
removal cost <0.2%; random twin better at 32 and 96 steps). Source-level
audit (`model.py:166-189`) confirmed no graph-level skip: the bypass was
learned, functional seam-to-seam identity — the target of this campaign.

## 6–9. Preregistration, architecture, and bypass audit

Gates were frozen in `campaign_04_preregistration.md` before training:
A integrity, B pretrained ≥2% better than matched twin, C donor
removal/neutralization ≥2% worse, D language ≤1.50× host and macro ≤5.7396,
E seed replication. Candidates: **delta-attention** (full-width seams +
donor-delta), **bottleneck-128 attention** (893,696 trainable), and
**bottleneck-128 + delta**. The delta pathway makes an identity donor
contribute exactly zero (unit-tested); internal Qwen residuals were left
intact as part of the pretrained computation.

## 10–13. Matrix, budgets, resources

Eight real-parent runs total across three sequential Actions jobs
(screen 1698 s including one preserved failed attempt whose training
completed but whose reporting hit a fixed accounting defect; twins
1565 s; replication 1101 s). Each run: 32 AdamW steps, 2,160 supervised
tokens, seed-matched data, CPU float32. Parents frozen and
fingerprint-verified throughout.

## 14–19. Results (development macro token-NLL, lower is better)

| Candidate | Pretrained | Random twin | Donor removed | Donor zeroed | Gate B | Gate C | Gate D |
|---|---|---|---|---|---|---|---|
| Delta-attention | 7.7019 | 7.8269 | 9.2264 | 13.8312 | FAIL (+1.6%) | PASS | FAIL |
| Bottleneck-128 | 8.0198 | 7.7549 | 8.0252 | 47.8779 | FAIL | FAIL | FAIL |
| Bottleneck-128 + delta | 7.7873 | 8.0091 | 15.3840 | 42.3203 | **PASS (+2.8%)** | **PASS** | FAIL |
| Bottleneck+delta, seed 123 | 8.0879 | 8.2268 | — | — | +1.7% (direction kept) | — | FAIL |

Host language NLL 4.5595; all candidates' language 1.82–1.85× host.
Donor relative update remained ≈0.08 (pretrained) vs ≈0.015 (random):
the donor's signal is small, but under forcing the pretrained version of
that small signal is worth measurably more than a random one.

## 20–24. Replication, integrity, failures, held-out status

Seed-123 replication preserved the benefit's direction. Parent
fingerprints unchanged in every job; bridge-only checkpoints reconstruct
(identical logits in unit tests; saved reference hashes in
`campaign_04_checkpoint_integrity.json`). One failed screen attempt was
preserved with its failure records. **Held-out confirmation:
NOT_RUN_NO_PROMOTION** — Gate D failed for every candidate.

## 25. Tests and CI

227 offline tests pass; lint/format clean; final-head CI verified in PR.

## 26. Plain-English explanation

In Campaign 3 the inserted foreign chapters could be deleted without the
reader noticing. In Campaign 4 we rewired the book so the story *must*
pass through those chapters — and now deleting them genuinely damages it,
and blank replacement chapters do slightly worse than the real ones at two
independent attempts. That is the first real evidence the transplant
contributes something. But the rewired book is still a bad book: forcing
the story through two foreign chapters at a narrow doorway loses too much
of the original. Contribution: demonstrated, weakly. Useful inheritance:
not yet.

## 27. What has been learned

Neither sequence context (C3) nor structural forcing (C4) is sufficient.
The donor's usable signal at this boundary is intrinsically weak (two of
28 blocks; ~8% relative update), so the interface has too little to work
with no matter how it is constrained.

## 28. What should happen next

A preregistered **donor-placement/sector-width study under the
mandatory-use (delta) architecture**: vary which and how many Qwen blocks
are transplanted (still whole blocks, still frozen), with the same twins,
ablations, and gates. If a wider or better-placed sector does not produce
a larger pretrained-specific benefit at acceptable performance, the
evidence will support concluding that this donor/host pairing at this
scale does not yield useful inheritance — a defensible final negative.

Full evidence: `campaigns/campaign_04/` (audit, preregistration, bypass
audit, registries, per-run records, results, controls, ablations,
activation diagnostics, training log, integrity, resource and recovery
reports). Candidate binaries remain in named GitHub Actions artifacts;
no parent weights are in Git.
