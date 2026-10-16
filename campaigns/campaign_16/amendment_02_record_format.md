# Campaign 16 — Amendment 02 (2026-10-16, before attempt-2 results)

Failure F-C16-002 destroyed the score evidence of attempt 1 after
all 18 runs had completed training (salvaged registries prove
completion; no primary scores were inspected — only completion
statuses and bundle metadata). Corrective change, confined to
record format: the runner's saved training trace is compacted
(indices, batch loss, global gradient norm, seam-aggregated
gradient norms) instead of embedding per-parameter gradient
dictionaries for all 96 steps. Training mathematics, conditions,
learning rates, seeds, budgets, evaluation, and decision rules
are unchanged from the frozen preregistration. Panels rerun
sequentially as attempt 2 to avoid concurrent posting storms.
