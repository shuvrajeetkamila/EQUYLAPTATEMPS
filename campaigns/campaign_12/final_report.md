# Campaign 12 — Final Report: The Structure-Scale Test

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Question:** Does the intact donor sector beat scrambled versions
of itself at two scrambling granularities — within-row (B) and
full-tensor (C) — reproducibly across seeds and budgets, with a
random twin (D) as reference?
**Classification: Outcome C** — controls valid; no reliable
learned-structure advantage at either granularity.
**Promotion:** false. **Held-out:** `NOT_RUN_NO_PROMOTION`.

## Design

Same frozen pairing/interface as Campaigns 10–11 (Mamba-130M host;
SmolLM-135M sector [24,30); cut [11,13); `natgeom_in_trained_out`).
20 fresh runs: 4 conditions × seeds {42, 123, 31415} at 32 steps,
plus 4 × {42, 123} at 96 steps. B permutes each weight matrix
within its rows (row statistics exactly preserved); C is Campaign
11's full-tensor permutation, reproduced bit-identically (verified
by SHA-256 on all 54 tensors). Preregistered gate: intact must
beat *both* B and C by ≥2% (seed 42) and ≥1% (seeds 123, 31415).

## Results (macro development NLL, lower better; benefit favors intact)

**32 steps (primary):**

| Seed | A intact | B row | C full | D random | vs B | vs C | vs D |
|---|---|---|---|---|---|---|---|
| 42 | 8.3670 | 8.8836 | 8.6061 | 8.8099 | +5.82% | +2.78% | +5.03% |
| 123 | 8.3627 | 8.4970 | 8.3152 | 8.3982 | +1.58% | −0.57% | +0.42% |
| 31415 | 8.7332 | 8.4337 | 8.3138 | 8.8103 | −3.55% | −5.04% | +0.87% |

**96 steps (diagnostic):**

| Seed | A | B | C | D | vs B | vs C | vs D |
|---|---|---|---|---|---|---|---|
| 42 | 7.8392 | 7.7717 | 7.6646 | 7.7089 | −0.87% | −2.28% | −1.69% |
| 123 | 7.5806 | 7.7530 | 7.6519 | 7.5696 | +2.22% | +0.93% | −0.15% |

Descriptive primary means (n=3, no significance claim): vs B
+1.28% (range −3.55…+5.82), vs C −0.95% (−5.04…+2.78), vs D +2.11%.

## Validity and diagnostics

All 10 scrambled-control runs pass every frozen validity criterion
(exact multisets incl. per-row for B; fingerprints changed; trained
graft updates 0.57–0.96, well above the 0.05 floor; ablations
finite and far above intact; parents unchanged). The random twin
remains near-silenced (update ≈0.001), so intact's positive vs-D
numbers carry no donor-specificity weight (Campaign 9 lesson) and
the claim rests on B/C as preregistered. Ablation ordering
bypass<zero holds in 10 of 12 scrambled runs; the two exceptions
(bypass worse than zero, both far above intact) are reported per
Amendment 01. Usefulness gate G-C fails at every seed (A language
ratios 1.88–1.92× vs 1.50× ceiling; macros 8.36–8.73 vs 6.00).

## Interpretation

The seed-42 pattern that drove Campaigns 9–11 (intact ahead of its
controls) appears again at seed 42 — and again fails to replicate.
At seed 123 the full-tensor control already matches the intact
sector; at the new seed 31415 *both* scrambled controls beat the
intact sector clearly, driven by the intact run's own worst seed
(8.7332) while controls stay in their usual band (8.31–8.43). At 96
steps the picture is mixed-to-negative. There is no scrambling
granularity at which learned arrangement shows a reproducible
advantage: whatever the intact sector contributes beyond an
arbitrary injected sector is carried by weight statistics/scale
and seed-level training variance, not by fine or coarse learned
arrangement — at this pairing, placement, interface, and budget.

This strengthens Campaign 11's Outcome C with an independent seed
and a second control granularity, and it closes the "maybe the
shuffle was too destructive" objection in both directions: the
gentler within-row shuffle (row statistics intact) is beaten by
the intact sector only at seed 42, and the harsher full-tensor
shuffle matches or beats it at two of three primary seeds.

## Limitations

Three seeds per cell is small; differences of ±0.3–0.5 NLL between
seeds of the *same* condition show seed variance is large relative
to any condition effect. One pairing, one sector, one interface,
short budgets. The claim is narrow: no reproducible
arrangement-specific advantage was found under these conditions —
not that cross-family transfer is impossible.

## Next scientifically justified experiment (not authorized)

If pursued: increase seed count for the intact-vs-full contrast
only (cheapest decisive cell), or test whether *any* pairing shows
arrangement sensitivity by applying the same two controls to a
same-family (transformer→transformer) transplant as a positive
control for the method itself. Either requires a new
preregistration.

## Provenance

Jobs `38726922262` (32-step), `38738479882` (96-step), validation
`38726596264`; bundles SHA-256-verified. Preregistration frozen at
`3f5df5f`. One amendment (validity-criterion implementation),
disclosed in `implementation_audit.md`. All 20 runs reported;
none rerun or excluded.
