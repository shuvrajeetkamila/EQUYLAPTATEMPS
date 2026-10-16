# Campaign 16 — Campaign 15 Reanalysis (from raw records)

Recomputed independently of the published summaries:

| Quantity | Value |
|---|---|
| Step-0 delta (old−new), mean | +1.9234 (new starts better, 3/3) |
| Step-32 deltas | −1.0046, −0.8541, −1.1111 (old better, 3/3) |
| Step-96 delta, mean | +0.1575 (new better, 3/3, small) |
| Baseline LR | 3e-4 (frozen registry) |
| Median pre-clip gradient norm, steps 1–32 | old ≈ 12.9–14.1; new ≈ 10.6–12.1 |
| Clipping frequency (norm > 1.0) | 100% of steps, both conditions |

**Hypotheses registered for Campaign 16:** (H1) learning-rate
scale: the new connector's smaller early gradients make the
shared LR effectively smaller for it; raising LR should
relatively help the new connector early. (H2) parameterization
geometry independent of LR: rankings stable across the sweep.
(H3) evaluation noise at step 96: small late differences may not
survive the sweep's replication. The frozen decision rules
(`decision_rules.md`) map sweep outcomes onto these without
post-hoc freedom.
