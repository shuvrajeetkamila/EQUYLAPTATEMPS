# Campaign 17 — Final report: Native-Depth Placement Redesign

**Primary label: `INTERFACE_REPAIRED_STRUCTURAL_DAMAGE_REMAINS`.**
Held-out: `NOT_RUN_NO_PROMOTION`. No training was preregistered or run.
Evidence: GitHub Actions run `38899506528` (commit `5a8ebf4`),
SHA-verified bundle in `diag_38899506528/`.

## Answers to the campaign questions

| Question | Answer |
|---|---|
| Does exact reconstruction pass? | **Yes** — Mode B (historical path, sector replaced by itself, seams bypassed) reproduces the native model bitwise: Δmacro 0.0, logit max/mean abs diff 0.0, 270/270 tensors identical. |
| Does identity replacement through the new interface pass? | **Yes** — Mode C (same checkpoint's blocks 11–12 re-inserted at their own position through the placement engine, residual-identity seams) is likewise bitwise identical to native on every frozen tolerance. |
| Are block count and depth preserved? | **Yes** — 30 → 30 executed blocks in every engine mode; total parameters equal native (134,515,008) plus, for Mode C only, the 667,008 identity-initialized seam parameters. |
| Any accidental deletion, duplication, or reordering? | **None.** Manifests record zero removed, zero duplicated, and moved blocks exactly equal to the declared exchanges. The placement engine makes deletion/duplication unrepresentable (non-permutations are rejected before assembly). |
| Exchange result? | **D1** (sectors [11,13) ↔ [24,26) exchanged): macro 11.1684, **+8.3953** vs identity — substantial damage (> 1.0). **D2** (adjacent blocks 15 ↔ 16 swapped): macro 2.9279, **+0.1547**. |
| Native score? | **2.773126** (macro dev NLL), repeated exactly; matches the Campaign 14–16 reference 2.7731 within tolerance. |
| Remaining damage after removing historical structural mistakes? | Reconstruction damage 0.0; placement-interface damage 0.0. Everything that remains is **structural-exchange damage**: putting correct blocks in the wrong depth positions destroys function even with perfect interfaces and identical tensor values. |
| Proven findings | The pipeline can now preserve native depth, block identity, order, and function exactly (engineering claim, proven). Moving native blocks to different depths is itself severely damaging; damage grows with exchange distance (adjacent swap +0.15 vs cross-region sector exchange +8.40). |
| Uncertain explanations | *Why* distant exchange is catastrophic (position/RoPE phase mismatch vs learned depth specialization) is not separated by this campaign; both remain hypotheses. |
| Gates | A pass, B pass, C pass, D measured as frozen (D1 substantial), E pass, F **fail** (D1 +8.40 > 0.5 readiness ceiling). |
| Cross-family readiness? | **Not ready** under the frozen rule: readiness required D1 ≤ 0.5. The placement *machinery* is validated; the structural sensitivity says a naive matched-depth foreign graft would likely fail for positional/functional reasons, so a pilot must be designed around that knowledge, not around optimism. |
| Single most defensible next step | A Campaign 18 preregistration that treats exchange damage as the object of study: first map how damage scales with exchange distance and position within the *same* model (cheap, fully controlled), and only then design a cross-family pilot whose placement/interface choices are informed by that map. |
| Pushed? Exact SHA? CI? PR? | See `publication_verification.md` (filled after push verification). PR #1 remains open and unmerged. Held-out untouched. |

## Mode table (macro dev NLL, lower is better)

| Mode | Macro | Δ vs native | Δ vs identity (C) | Logit max/mean vs native | Tensors identical |
|---|---|---|---|---|---|
| A native | 2.773126 | — | — | — | — |
| B historical reconstruction | 2.773126 | 0.000000 | 0.000000 | 0.0 / 0.0 | 270/270 |
| C identity replacement (new engine) | 2.773126 | 0.000000 | — | 0.0 / 0.0 | 270/270 |
| D1 sector exchange [11,13)↔[24,26) | 11.168408 | +8.395283 | +8.395283 | 44.2 / 9.52 | 270/270 |
| D2 adjacent swap 15↔16 | 2.927872 | +0.154746 | +0.154746 | 14.5 / 1.15 | 270/270 |

## What this proves and does not prove

It proves the *engineering*: native-depth, identity-preserving
placement with exact functional reconstruction is achievable in this
pipeline, and the historical depth-changing defect (30→34 blocks,
blocks 11–12 deleted, sector 24–29 duplicated) is eliminated by
construction in the new engine. It does **not** prove anything about
cross-family transfer; no foreign weights were used, and the frozen
readiness gate failed on structural grounds. The transplant
hypothesis remains unproved.

## Damage attribution (no double counting)

Reconstruction 0.0 · placement interface 0.0 · structural exchange
+8.3953 (D1) / +0.1547 (D2) · connector damage not measured in this
campaign by design (Campaign 14 evidence stands separately) ·
training effects none · unexplained 0.0. See
`damage_attribution.json`.

## Artifacts

`diag_38899506528/` (raw evidence) · `placement_manifest.json` ·
`structural_comparison.csv` · `evaluation_results.csv` ·
`baseline_reference.json` · `reconstruction_results.json` ·
`identity_replacement_results.json` ·
`matched_depth_exchange_results.json` · `damage_attribution.json` ·
`run_manifest.json` · `run_registry.json` · `failure_registry.json`
(one infrastructure trigger-lag entry, F-C17-001; the frozen job ran
exactly once) · `results_raw.jsonl` · `environment_manifest.json` ·
`implementation_audit.md` · `checksums.sha256` · `test_report.txt`.
