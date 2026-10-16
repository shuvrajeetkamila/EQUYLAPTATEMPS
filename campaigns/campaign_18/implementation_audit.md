# Campaign 18 — Implementation audit

- `scripts/campaign18_map.py`: single runner with `--set map|heldout`
  (controls, invalid-manifest probes, inverse controls, dev/val or
  held-out conditions, mechanism subset), reusing the Campaign 17
  engine, frozen panel, and bundle mechanism. Every condition
  records manifest, per-block hash audit, per-example scores.
- `scripts/fit_campaign18_predictor.py`: development-only OLS fit,
  validation selection, writes the frozen predictor and the recorded
  held-out opening; never reads held-out outcomes.
- `scripts/publish_campaign18.py`: derives all published artifacts
  from the two SHA-verified bundles; applies frozen gates and label
  precedence mechanically.
- Inverse-operation interpretation is documented in
  `placement_engine_validation.json` (stateless engine; composition
  to identity + exact identity control).
- Failures: none. Both Actions runs succeeded first time; the frozen
  dev/val job ran once and the sealed held-out job ran once.
