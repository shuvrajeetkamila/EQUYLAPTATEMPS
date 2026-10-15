# Campaign 12 — Preregistration (frozen before confirmatory training)

**Project:** HETEROGENEOUS AI MODEL SURGERY
**Registered:** 2026-10-14
**Question:** Does the amount and type of structural scrambling of a
transplanted donor sector affect the child, under identical
statistics, interface, activity measurement, budget, and seed?

## Hypothesis H12

The intact pretrained donor sector (A) outperforms both
statistics-preserved scrambled versions of itself — within-row
shuffled (B) and full-tensor shuffled (C) — reproducibly across
seeds at the primary budget. If A does not reliably beat B and C,
the Campaign 9–11 donor-specific effects are not attributable to
the donor's learned fine-grained weight arrangement at either
scrambling granularity tested.

## Frozen setup (identical to Campaigns 10–11 unless stated)

- Recipient: `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
- Donor: `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`
- Recipient cut [11, 13); donor sector [24, 30); donor frozen;
  interface `natgeom_in_trained_out` (donor-native geometry in,
  trained output seam); macro development NLL (lower better).
  Host language dev NLL 4.559452953338623.
- Data, optimizer (AdamW lr 3e-4), batch, max length 512, training
  seed handling: exactly the Campaign 11 registry configuration.

## Conditions (all fresh runs in Campaign 12; no historical reuse)

- **A — intact:** donor sector as transplanted. Reference condition.
- **B — within-row shuffle (new):** for each eligible 2D matrix
  `[out, in]`, entries permuted independently within each output
  row. Preserves every row's exact value multiset (hence row
  mean/variance/L1/L2 norms and the tensor multiset); destroys the
  input-coordinate arrangement within each output unit. Shuffle
  seed **20241014** (one fixed seed; no permutation shopping).
- **C — full-tensor shuffle:** Campaign 11's
  `shuffled_sector_control`, unchanged, seed **20241014**.
  Reproduction of the Campaign 11 control inside this campaign.
- **D — random twin (reference):** established random-sector
  control, `random_sector_seed` **1234**, same interface treatment.
- Eligibility (B and C): the 42 two-dimensional donor-sector
  matrices (7 per block × 6 blocks). 1D RMSNorm gains (12) are
  excluded and preserved, as in Campaign 11. Bypass/zero ablations
  run inside every run (diagnostic condition E, not a primary
  contrast).

## Runs (20 confirmatory)

- Primary: 32 steps × seeds {42, 123, 31415} × conditions
  {A, B, C, D} = 12 runs (`c12_{pre,row,full,rand}_{seed}`).
- Budget diagnostic: 96 steps × seeds {42, 123} × 4 conditions
  = 8 runs (suffix `_96`). The 96-step panel is diagnostic: it is
  reported in full and cannot change the classification, which
  rests on the 32-step primary (as frozen here, before any run).

## Control validity (frozen; checked per run from its records)

For B and C runs: (1) every eligible tensor reports exact multiset
preservation (B additionally: every row's multiset preserved);
(2) every eligible tensor's fingerprint changed; (3) trained graft
relative update ≥ 0.05 (activity floor, from Campaign 11);
(4) bypass and zero ablation macros finite and above intact;
(5) `parent_weights_unchanged` true. If any B or C run fails,
that condition's comparisons are invalid and the outcome is D
unless a dated amendment filed before inspecting outcomes defines
a repair. D is a reference: its activity is reported, and beating
an inactive D is not donor-specificity evidence (Campaign 9
lesson); A's donor-specificity claim rests on B and C.

## Primary contrasts and gates (frozen)

Benefit vs control X = 100·(L_X − L_A)/L_X at matched steps+seed.
**Learned-structure gate:** A beats *each* of B and C by ≥2% at
seed 42 and ≥1% at seeds 123 and 31415 (32 steps). This extends
the canonical per-seed thresholds (≥2% @42, ≥1% @123) to the new
seed and the new within-row control — a conservative gap-fill
recorded here before any result (the canonical protocol predates
both). **Usefulness gate G-C (canonical, unchanged):** A's
language ratio ≤ 1.50× host and macro ≤ 6.00 at all three primary
seeds. **Promotion:** learned-structure gate + G-C + all controls
valid + causal ablation support (A intact < bypass) + integrity;
only then may held-out open once. Otherwise held-out =
`NOT_RUN_NO_PROMOTION`.

## Frozen outcome mapping (exactly one; Campaign 11 definitions)

- **A:** learned-structure gate and G-C pass (promotion-eligible).
- **B:** learned-structure gate passes; G-C fails. (Donor-specific
  but not useful.)
- **C:** controls valid; learned-structure gate fails — no
  reliable advantage of the intact donor over its scrambled
  selves, at either granularity.
- **D:** a B/C control invalid, or other documented validity
  failure — inconclusive.

## Statistics and reporting

Per contrast: all seed-level benefits, plus descriptive mean,
median, range over the three primary seeds (n=3; descriptive
only, no significance claims, no token-level replication). Ties
(|benefit| < 0.1%) reported as ties. Every run reported,
including FAILED/BLOCKED with reasons; no selective reruns.
Interrupted jobs resume by re-triggering the same phase; the
runner skips verified-complete runs and never reruns a completed
run for a better score. Raw per-run records, manifests, and
SHA-256 checksums published; the verdict is computed mechanically
by `scripts/publish_campaign12.py`. TEST data never used.
