# Campaign 8 — Host-Statistics Interface Calibration

**Question:** does distribution-aware interface calibration
(`covariance_orthogonal`, TRAIN-statistics only) give the frozen SmolLM
[24,30) sector a reproducible advantage over its matched random twin in the
Mamba host — at 32 steps, at 96 steps, and across seeds? Pairing, protocol,
metric, and gates are identical to Campaign 7; only interface initialization
changes. **Status:** preregistration frozen 2026-10-14; Stage D screen pending.

Key files: `implementation_audit.md` (verified C7 forward path),
`preregistration.md`/`.json` (frozen design and gates),
`interface_statistics_baseline.json` + `interface_statistics_report.md`
(Stage B diagnostics), `experiment_registry.json` (Stage D runs). Raw results
and the final report (`HETEROGENEOUS_AI_MODEL_SURGERY_CAMPAIGN_08.md` at repo
root) are added as stages complete. Held-out: `NOT_RUN_NO_PROMOTION` unless
the frozen promotion rule passes.
