# Campaign 10 Awake-Control Design (frozen rule)

## Why the Campaign 9 twin slept

Under donor-native coloring, the twin's graft relative update was ≈0.001
(intact ≈ bypass): the random blocks' residual-branch outputs are tiny
relative to the large colored residual stream, so the sector acted as an
identity pipe. Campaign 10's condition D rescales the random sector so it
is *demonstrably active* before training — using only label-free activation
measurements on TRAIN text.

## Frozen calibration rule (deterministic, no outcome peeking)

- **Activity metric** (established `graft_delta` formula): mean over the 8
  TRAIN statistic windows of `||graft_out − graft_in||₂ / ||graft_in||₂`,
  measured at initialization (after interface init, before any training).
- **Target** `u*`: the native donor sector's own relative update, measured
  identically on the same windows (frozen donor parent, donor tokenizer,
  sector [24,30) input vs output). A fixed architecture-level quantity.
- **Adjustment:** multiply every graft block's residual-output projections
  (`self_attn.o_proj.weight`, `mlp.down_proj.weight`) by a single scalar
  `s`, estimated as `s = u*/u_measured`, clipped to **[0.05, 100]** (the
  programme's established safety bounds). Re-measure once; if the achieved
  activity is outside **[0.5, 2.0] × u***, apply one correction
  (`s ← clip(s · u*/u_achieved)`) and re-measure.
- **Success criterion:** achieved activity inside the band, all values
  finite. Otherwise record `AWAKE_CALIBRATION_FAILED`; D is then flagged
  `INVALID_CONTROL` in the analysis (never silently treated as awake).
- Calibration touches only the twin's frozen random weights (scaling is
  recorded; weights remain `requires_grad=False`); it uses no labels, no
  dev/held-out data, and no outcome information. Condition C is the same
  twin without this scaling, isolating the calibration's effect.
