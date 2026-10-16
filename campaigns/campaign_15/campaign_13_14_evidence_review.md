# Campaign 15 — Campaign 13/14 Evidence Review

Recomputed from SHA-verified raw records during the Campaign 15
audit (see `repository_audit.md` for method).

| Quantity | Value | Source |
|---|---|---|
| Native recipient macro dev NLL | 2.7731 | C14 diagnostics V0 (also C13 jobs) |
| Native repeat difference | 0.0 | C14 V0 first vs second |
| Native-placement reconstruction (V1/V3) | 2.7731, logit Δ 0.0, 273/273 tensors | C14 diagnostics |
| Old seams at native placement (V2) | 24.4656 (+21.69) | C14 diagnostics |
| C13 placement, neutral seams (V5) | 12.0150 (+9.24) | C14 diagnostics |
| Historical configuration untrained (V6) | 14.2481 (+11.47) | C14 diagnostics |
| Trained 32 steps, new connector (V7) | 8.4623 | C14 trained job |
| Trained 32 steps, old connector (V8) | 7.4578 | C14 trained job |
| Campaign 13 F (same configuration as V8) | 7.4577 | C13 self-test job `38770694519` |

**Interpretation carried into Campaign 15:** initial damage is
fully decomposed (connectors ≫ placement at initialization), but
the single-seed trained inversion (old connector finishing ahead)
is unexplained. Hypotheses the three-seed panel can distinguish:
(i) seed noise around a real initialization benefit; (ii) the
historical initialization is a genuinely better starting point
*for optimization at this topology* despite damaging the initial
function; (iii) budget interaction (ranking changes by step 96).
The campaign tests these without changing the question midway.
