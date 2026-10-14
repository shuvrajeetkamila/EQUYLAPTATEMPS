# Campaign 9 Preregistration — Donor-Native Representation Geometry (FROZEN)

**Project:** HETEROGENEOUS AI MODEL SURGERY · **Frozen:** 2026-10-14, before
any Campaign 9 training. Pairing, sector [24,30), cut [11,13), data, protocol,
metric (macro dev NLL), and gates are identical to Campaigns 7–8; only the
interface initialization changes. Audit: `repository_audit.md`; geometry:
`representation_geometry.md`.

## Hypothesis H9

A foreign host cannot directly provide the representation geometry a
pretrained donor block expects. Initializing the interfaces to explicitly
reconstruct the donor's native input geometry (whiten child stream → color
with donor-native covariance; symmetric recipient-native coloring on output)
will improve the pretrained child **relative to an equivalently treated
random twin**. Alternatives (optimization difficulty, placement, objective)
are controlled by holding everything else fixed. H0: no reproducible useful
donor-specific advantage — the default unless gates pass.

## Frozen conditions

| # | Condition | Sector | Interface init | Status |
|---|---|---|---|---|
| A | Original | pretrained | `random` | stored C7/C8 records |
| B | Original | random twin (1234) | `random` | stored C7/C8 records |
| C | **Donor-native geometry** | pretrained | `donor_native_geometry` | **new** |
| D | **Donor-native geometry** | random twin (1234) | `donor_native_geometry` (identical matrices/procedure) | **new** |

Primary comparison: **C vs D, 32 steps, seed 42**, replicated at seed 123
(registered by amendment after the screen, before running, regardless of
direction). A 96-step C/D pair may follow by dated amendment only if the
32-step comparison completes validly. Trainable sets identical (two dense
seams, 888,768); statistics budget identical (8 TRAIN windows/child).

## Frozen gates (unchanged since Campaign 7)

- **G-B:** macro_pretrained ≤ 0.98 × macro_twin (same condition).
- **G-C:** language ratio ≤ 1.50 × host (4.5595) **and** macro ≤ 6.00.
- **Promotion:** G-B ∧ G-C at seed 42 **and** seed-123 replication benefit
  ≥ 1%; only then held-out opens, once. Else `NOT_RUN_NO_PROMOTION`.
- Relative improvement = (L_twin − L_pretrained)/L_twin × 100% (positive =
  pretrained better). Single primary comparison; no best-run selection;
  failed runs preserved; amendments dated and pre-filed.

## Frozen outcome mapping (exactly one)

- **A:** reproducible donor-specific advantage **and** usefulness gates pass.
- **B:** reproducible advantage (G-B replicated) but usefulness fails.
- **C:** valid experiment; no reliable useful donor-specific advantage.
- **D:** inconclusive — documented technical/resource/control-validity failure.

Known limitation, stated in advance: geometry matching corrects second-order
statistics only; a negative result rules out *this* geometry reconstruction,
not every conceivable architecture-aware interface.
