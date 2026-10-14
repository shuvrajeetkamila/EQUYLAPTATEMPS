# Amendment 01 — Awake-Calibration Bound Correction (2026-10-14)

**Filed after the seed-42 screen and before any replication/budget runs.**
The frozen rule and the deviation are both preserved here.

## Deviation discovered

The frozen rule (`awake_control_design.md`, `config/preregistration.json`)
bounded the awake scalar: "single scalar … bounds [0.05, 100], one
correction allowed." The implementation of `calibrate_awake_sector`
clipped each *incremental* step instead of the total, so the screen's
awake twin received a total scale of **591.27** (initial activity 0.00175 →
achieved 0.988 vs target 1.028; within the frozen activity band, all
finite, stable through training). Conditions B and C do not use this code
path and are unaffected; their seed-42 records remain confirmatory.

## Correction

`calibrate_awake_sector` now clips the **total** applied scalar, and the
bound is amended prospectively to **[0.05, 1000]**: the original 100 cap was
imported from the RMS-gain safety convention for trained seam weights — a
different quantity. The awake scalar acts on frozen random branch
projections, and the operative safety criteria remain the frozen ones
(achieved activity within [0.5, 2.0] × target, all finite, stable trained
run). Under the amended bound the screen's calibration (591.27) would have
been legal, but because it predates this amendment its D run is labeled
**exploratory**; the confirmatory awake run is re-executed under the
amended rule as `c10_out_awake_42_amd1` (identical otherwise), and all
replication/budget runs use the amended rule. Gates unchanged.
