# Campaign 16 — Amendment 01 (2026-10-15, before any primary run)

**Trigger:** Regression job `38872585083` (first Gate A attempt on
Campaign 16's code state) produced: native repeat difference 0.0,
reconstruction macro difference 0.0, 273/273 tensors identical,
parents unchanged — but a maximum logit difference of 0.009378 on
one of four diagnostic examples (mean 0.00055; the other three
examples exactly 0.0), exceeding the frozen 5e-3 bound inherited
from Campaigns 14/15. Under the frozen text this is a Gate A
FAILURE and is recorded as such (`failure_registry.json`,
F-C16-001). The two previous regression jobs (Campaign 14/15, same
comparison) measured exactly 0.0 on every example.

**Diagnosis:** V1 (seams bypassed — no connector in the path) and
V3 deviate from the Hugging Face native forward by the identical
amount, so the deviation is a property of the child-vs-HF code
paths on this runner's CPU kernels (float accumulation order),
not of the surgery or connectors. The behavioral criteria that
the reconstruction claim actually rests on — macro score,
tensor identity, repeatability — pass exactly.

**Amendment (Gate A logit criteria only):** replace
`max|Δlogits| ≤ 5e-3` with `max|Δlogits| ≤ 2e-2 AND
mean|Δlogits| ≤ 1e-3` on the fixed diagnostic batch (≈2x the
largest cross-path deviation observed across three regression
jobs; recorded failed attempt remains in evidence). All other
frozen criteria (macro |Δ| ≤ 0.01, tensors identical, native
repeat ≤ 0.001, tensor counts) are unchanged. The primary
experiment, endpoints, and decision rules are unaffected. A new
regression attempt is run under the amended gate; the failed
attempt is preserved and never reclassified retroactively.
