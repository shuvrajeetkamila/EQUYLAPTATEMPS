# Campaign 15 — Frozen Decision Rules

Metric: macro dev token NLL (lower better). For each seed s:
`delta_t(s) = score_old(t, s) − score_new(t, s)` at step t;
positive favors the new connector.

Apply in this precedence order:

1. **IMPLEMENTATION_FAILURE** — Gate A (native reconstruction
   regression) fails, or any integrity breach (parents changed,
   wrong trainable set, resume-hash mismatch) invalidates the
   panel.
2. **INCONCLUSIVE** — any of the six primary runs is missing or
   INVALID under the preregistration (e.g., nonfinite loss,
   incomplete to step 96 without a valid resume), so the frozen
   rules cannot be applied completely.
3. **TRAINING_DEPENDENT** — the step-32 picture does not survive
   the duration check: sign(mean delta_32) ≠ sign(mean delta_96)
   (both nonzero), or the 3/3 consistency at step 32 becomes 3/3
   consistency in the opposite direction at step 96.
4. **REPAIRED_START_ADVANTAGE** — Gate B passes **and**
   delta_32 > 0 for all three seeds.
5. **INITIALIZATION_ONLY** — Gate B passes but Gate C does not
   (the new connector starts better but does not consistently win
   after 32 steps of training).
6. **NO_RELIABLE_ADVANTAGE** — Gate B fails and Gate C does not
   hold (no reliable advantage established for the new connector;
   direction of any differences reported as measured).

Reported statistics (descriptive; n=3, no significance claims):
per-seed deltas at 0/8/16/32/64/96; mean, median, standard
deviation of delta_32; seed win counts; step0→step32 improvement
per condition; a paired t-interval (df=2) reported with its width
as an uncertainty illustration only.

**Gate E recommendation mapping:** REPAIRED_START_ADVANTAGE →
preregistered cross-family pilot may be designed (Campaign 16
readiness, report only). INITIALIZATION_ONLY / TRAINING_DEPENDENT
→ focused optimization-dynamics diagnosis next.
NO_RELIABLE_ADVANTAGE → smallest distinguishing experiment.
IMPLEMENTATION_FAILURE/INCONCLUSIVE → fix correctness/
instrumentation before any scientific claim.
