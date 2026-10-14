# Amendment 01 — Calibration Scale-Gain Guard (2026-10-14)

**Filed after the frozen Stage D screen failed to execute and before any
amended run.** The original preregistered experiment is preserved and
distinguished below.

## What happened

Frozen Stage D (job `38561906471`, imported as `screen_38561906471/`) failed
both runs **before training**: `initialize_conditioned("covariance_orthogonal")`
raised `ValueError: Unsafe observed scale calibration`. Cause, from the stored
statistics: the input gain is `donor_native_input_rms / child_donor_input_rms
= 196.50 / 1.056 ≈ 186`, outside the Campaign-2-era safety range (0.05, 100).
The guard was written for the Qwen donor, whose native stream is O(1); SmolLM's
late-layer residual stream is genuinely ~196 RMS. The failure records are
preserved in the imported bundle; no training occurred, so no result was
contaminated.

## The amendment

New method `covariance_orthogonal_noscale` (in `alignment.py`, with tests):
identical orthogonal pair + covariance whitening, but a scale gain outside
(0.05, 100) is **skipped and recorded** instead of raising; in-range gains are
still applied. Justification, recorded in the Stage B report before this
failure was known: every donor block begins with per-token RMSNorm, which
cancels pure input scale — the gain is the least consequential component,
while whitening targets the operative directional/covariance mismatch. The
original `covariance_orthogonal` behavior is unchanged for historical runs.

Amended Stage D runs (new IDs, `experiment_registry_stageD_amended.json`):
`c08_calnoscale_pre_42`, `c08_calnoscale_rand_42` — identical in every other
respect (same statistics procedure for pretrained and twin). Gates, metric,
budgets, and outcome mapping are **unchanged**. In the final report, amended
results are labeled as the amended calibration; the frozen method's
non-execution is reported as a finding about method portability, not hidden.
