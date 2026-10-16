# Campaign 11 — Final Report: The Learned-Structure Test

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Question (H11):** Does the intact pretrained donor sector outperform
a statistics-preserved, structure-destroyed version of *itself* —
same weights, same value distributions, arrangement scrambled — under
identical treatment? If yes, earlier donor-specific effects reflect
learned organization. If no, they are explained by weight statistics
and scale alone.
**Classification: Outcome C** — valid control, no reliable
learned-structure advantage.
**Promotion:** false. **Held-out:** `NOT_RUN_NO_PROMOTION`.

## Design in one paragraph

Same pairing as Campaigns 7–10 (Mamba-130M host, SmolLM-135M donor
sector [24,30), cut [11,13), donor frozen, `natgeom_in_trained_out`
interface). Condition A is the stored Campaign 10 intact-pretrained
result. Condition B replaces the donor sector's 42 weight matrices
with within-tensor permutations of themselves (one frozen shuffle
seed, 20241014): every tensor keeps its exact multiset of values —
mean, variance, norms, quantiles identical — but the learned
arrangement is destroyed. Everything else (interfaces, data, steps,
optimizer, evaluation) is identical.

## Control validity (all frozen criteria met)

| Check | shuf_42 | shuf_123 | shuf_42_96 |
|---|---|---|---|
| 42/42 tensors, multisets preserved | ✓ | ✓ | ✓ |
| All eligible tensors actually permuted | ✓ | ✓ | ✓ |
| Trained graft relative update ≥ 0.05 | 0.561 | 0.592 | 0.565 |
| Ablations finite, intact < bypass < zero | ✓ | ✓ | ✓ |
| Parent weights unchanged | ✓ | ✓ | ✓ |

The control is **not** a silenced sector: the permuted graft is used
by the child about as much as the intact graft was (updates ≈0.56–
0.59, vs 0.48–1.34 for intact), so H11 was genuinely tested.

## Results (macro development NLL, lower is better)

| Steps | Seed | A intact | B shuffled | Intact benefit | Gate |
|---|---|---|---|---|---|
| 32 | 42 | 8.3665 | 8.6808 | **+3.62%** | ✓ (≥2%) |
| 32 | 123 | 8.3132 | 8.3098 | **−0.04%** | ✗ (≥1%) |
| 96 | 42 | 7.8392 | 7.7141 | **−1.62%** | secondary |

Benefit = 100·(L_B − L_A)/L_B. Reference: the Campaign 10 random
twin scored 8.8099 / 8.3945 at 32 steps — the shuffled sector sits in
the same band as both the intact sector and the random twin.

## Interpretation

At seed 42 the intact sector beats its scrambled twin modestly; at
seed 123 the two are indistinguishable (difference 0.003 NLL); at 96
steps the scrambled control is *ahead*. Under the frozen outcome
mapping this is **Outcome C**: there is no reliable, seed-consistent
evidence that the donor's learned arrangement — as opposed to the
statistics and scale of its weights — drives the donor-specific
effects seen in Campaigns 9–10. Those effects should now be read as
fragile and substantially explained by weight value statistics/scale
interacting with the geometry-aware interface, not by transferable
learned structure in the transplanted blocks. Usefulness gates also
fail throughout (best language ratio 1.80× vs the 1.50× ceiling;
best macro 7.71 vs 6.00), so no promotion and no held-out evaluation.

## What this does not show

It does not show donor pretraining is irrelevant in general — only
that, for this pairing, placement, interface, and budget, scrambling
the donor's internal arrangement costs nothing replicable. It also
does not rehabilitate the random twin: at seed 123 intact, shuffled,
and random are all within ~0.09 NLL of each other, i.e. the graft's
content matters far less than the interface treatment around it.

## Provenance

Screen job `38713676113`, budget job `38715108842`; both bundles
SHA-256-verified on import. Condition A: stored Campaign 10 records.
Preregistration frozen at `fb90ebb` before any Campaign 11 training;
Amendments 01 (trigger mechanism) and 02 (budget-run authorization)
are infrastructure/scheduling only. Full numbers:
`results.json`, `comparison_table.csv`, `control_validation.json`.
