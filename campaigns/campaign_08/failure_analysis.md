# Campaign 8 Failure Analysis

1. **Frozen calibration could not execute (main incident).** Both frozen
   Stage D runs failed pre-training: the `covariance_orthogonal` safety guard
   (gains must lie in (0.05, 100), written for the Qwen era) rejected the
   measured 186× SmolLM RMS ratio. Handled per the rules: failure preserved
   (`screen_38561906471/`), dated Amendment 01 filed **before** any amended
   run, new method `covariance_orthogonal_noscale` implemented and tested,
   amended runs given new IDs. The original method is untouched.
2. **No experiment failures after the amendment.** All six amended/Stage E/F
   runs COMPLETED with integrity verified.
3. **Scientific "failure" is the result, not an error:** calibration helped
   only the random twin. The 2×2 design prevented misreading this as donor
   progress — had we run the calibrated pretrained child alone against its
   uncalibrated self at some budgets, a misleading narrative would have been
   available. This is the campaign's methodological point.
4. Cosmetic: registry state `campaign: 2` field (see `integrity_report.md`).
