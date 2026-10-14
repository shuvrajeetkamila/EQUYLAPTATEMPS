# Campaign 8 Preregistration — Host-Statistics Interface Calibration (FROZEN)

**Project:** HETEROGENEOUS AI MODEL SURGERY  
**Frozen:** 2026-10-14, before any Campaign 8 training. Pairing, sector, data,
protocol, and metric are identical to Campaign 7 (see
`implementation_audit.md`); only the interface **initialization** changes.

## Hypotheses

- **H1 (calibration):** distribution-aware interface initialization
  (`covariance_orthogonal`: orthogonal variance-preserving pair + covariance
  whitening + residual-RMS matching, estimated on unlabeled TRAIN text only)
  improves the pretrained child relative to a matched random twin.
- **H2 (budget):** 96 steps may change the comparison; tested with matched
  budgets only, after Stage D.
- **H0 (null):** no reproducible useful calibration effect. H0 is the default
  conclusion unless the frozen gates say otherwise.

## Frozen conditions (primary comparison within each condition)

| # | Condition | Sector | Interface init | Status |
|---|---|---|---|---|
| 1 | Original | pretrained | `random` (C7) | stored C7 records (Stage A) |
| 2 | Calibrated | pretrained | `covariance_orthogonal` | **new** |
| 3 | Original twin | random (seed 1234) | `random` (C7) | stored C7 records (Stage A) |
| 4 | Calibrated twin | random (seed 1234) | `covariance_orthogonal` | **new** |

Primary endpoint: Conditions 2 vs 4 at **32 steps, seed 42**; replication at
seed 123 (Stage F). Stage E (contingent on Stage D completing with integrity
intact, regardless of direction): Conditions 2/4 and 1/3 at **96 steps**,
seed 42, registered by amendment before running. Trainable sets are identical
in all conditions (the two dense seams, 888,768 params); calibration consumes
the same 8 TRAIN statistic windows for pretrained and twin children alike.

## Frozen gates (identical to Campaign 7)

- **G-B:** `macro_pretrained ≤ 0.98 × macro_twin` within the same condition.
- **G-C:** `language_ratio_vs_host ≤ 1.50` (host dev 4.5595) and
  `macro_pretrained ≤ 6.00`.
- **Promotion:** G-B ∧ G-C for the calibrated condition at seed 42 **and**
  seed-123 replication benefit ≥ 1%. Only then may held-out be opened, once.
  Otherwise `NOT_RUN_NO_PROMOTION`.

## Frozen outcome mapping (brief §13, exactly one)

- **A:** calibrated pretrained beats matched controls reproducibly **and**
  meets usefulness criteria (held-out confirms if opened).
- **B:** reproducible donor-specific advantage (G-B replicated) but usefulness
  fails.
- **C:** valid experiment; no reliable useful pretrained advantage.
- **D:** inconclusive due to documented technical/resource limitation.

Statistics are diagnostics, never success evidence (Stage B report). Donor
dependence (bypass/zero) is reported but never gates. Failed runs preserved;
amendments dated and filed before the runs they affect; TEST never used for
selection.
