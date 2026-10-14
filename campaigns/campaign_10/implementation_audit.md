# Campaign 10 Implementation Audit

## Changes vs Campaign 9 (exactly two, both frozen in the preregistration)

1. **Output pathway:** new initialization `natgeom_in_trained_out`
   (`alignment.py`): the input seam receives Campaign 9's donor-native
   geometry identically; the output seam is restored to the generic
   constructed `DenseSeam` default and purely trained (record field
   `output_pathway: trained_default_init_no_geometry`). Verified by
   `initialization.json` in every run and by unit test.
2. **Awake twin (condition D):** `calibrate_awake_sector` (`controls.py`)
   rescales the random graft's residual-output projections to reach the
   native donor sector's relative-update target (measured: **1.0282** on
   the 8 TRAIN windows). Confirmatory runs: initial activity ≈0.0018 →
   achieved ≈0.98–0.99, total scale ≈586–591 (amended bound [0.05, 1000],
   Amendment 01), calibrated=true, all finite; post-training graft update
   ≈0.60 (still active, vs raw twin ≈0.001).

## Parity and integrity

All conditions: same child structure, **888,768 trainable** seam params
(awake calibration adds none), same data/optimizer/budgets; donor frozen —
`parents_unchanged: true` in all five jobs; random twins share seed 1234
(C and D are the same random blocks; D additionally rescaled). The one
deviation (per-step vs total bound clipping in the exploratory screen D)
is documented in `amendment_01_awake_bound.md`; the confirmatory D was
re-run under the amended rule (`c10_out_awake_42_amd1`).

## Activity/bypass behavior of the controls (important)

Raw twin C remains near-identity after training (update ≈0.001; intact ≈
bypass at 32 steps: 8.8099 vs 8.8090). Awake twin D is genuinely active
(update ≈0.60; bypass degrades it sharply, e.g. 9.3403 → 13.7901) — the
awake objective was achieved. Note D's activity is *destructive*: an
amplified random sector damages the child, which is why beating D is a
weaker result than beating an active-and-benign control would be.
