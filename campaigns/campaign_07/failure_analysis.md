# Campaign 7 Failure Analysis

**Experiment failures: none.** All four registered runs COMPLETED with verified
integrity. The negative result is a scientific finding, not a failure.

Process failures encountered and preserved:

1. **Probe v1 evidence loss (run `38540675630`):** the probe printed pinned
   revisions to stdout, but this sandbox cannot reach the Actions log/artifact
   download hosts (connection EOF). Fixed by posting probe output to PR #1 via
   the API (probe v2, run `38540795693` and later); no data was fabricated —
   configs/licenses come from the probe-2 PR comment.
2. **Replication workflow commit race:** `campaign07-replication.yml` was
   written in parallel with the commit that was meant to include it and stayed
   untracked; the first trigger commit therefore launched nothing. Detected by
   checking the workflow registry (404), fixed by committing the file before
   re-triggering. No experiment was affected.
3. **Zero-graft asymmetry (observation, not failure):** at seed 123 the
   pretrained child's zero-graft macro (10.211) is *better* than at seed 42
   (44.072) — after training, the seams partially compensate for a zeroed
   sector. Reported as-is; it does not change any gate (ablations are
   diagnostics under the frozen preregistration).
