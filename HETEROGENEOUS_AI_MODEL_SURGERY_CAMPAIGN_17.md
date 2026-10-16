# Heterogeneous AI Model Surgery — Campaign 17 summary

**Native-Depth Placement Redesign — COMPLETE.**
Label: `INTERFACE_REPAIRED_STRUCTURAL_DAMAGE_REMAINS`.

A new placement engine (`src/equylaptatemps/native_placement.py`)
assembles children whose executed blocks are an explicit permutation
of the host's 30 blocks — deletion and duplication are
unrepresentable. On the frozen dev panel (Actions run 38899506528):

| Mode | Macro NLL | Δ vs native |
|---|---|---|
| A native SmolLM-135M | 2.773126 | — |
| B historical-path reconstruction | 2.773126 | 0.0 (bitwise) |
| C native-depth identity replacement | 2.773126 | 0.0 (bitwise) |
| D1 sector exchange [11,13)↔[24,26) | 11.168408 | +8.3953 |
| D2 adjacent swap 15↔16 | 2.927872 | +0.1547 |

Placement/interface damage is now exactly zero; the remaining damage
is structural: native blocks only function at their native depths.
The frozen cross-family readiness gate (D1 ≤ 0.5) failed, so no
transfer claim is made. Full record:
`campaigns/campaign_17/final_report.md`.
