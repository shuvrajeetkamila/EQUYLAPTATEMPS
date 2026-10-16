# Campaign 19 — Validated run matrix explanation (Stage 2)

| ID | Purpose | Trained? | Seeds | Notes |
|---|---|---|---|---|
| D0 | Donor validity (donor alone vs in-body scrambled donor) | No | — | Gate; failure blocks all graft work |
| A | Untouched recipient baseline | No | — | Reference 2.773126 expected |
| B | Identity control through replacement pathway | No | — | Gate; must be exact |
| C | Intact foreign graft | Yes (connectors) | 42, 123, 31415 | Primary condition |
| D | Scrambled donor (within-tensor permutation) | Yes | 42, 123, 31415 | Same connectors/budget as C |
| E | Distribution-matched random donor | Yes | 42, 123, 31415 | Same connectors/budget as C |
| F | Ablation of trained C (`bypass_graft`) | No | per C | Removes donor+connectors (limitation recorded) |
| G | Restoration (ablation off) | No | per C | Must reproduce C exactly |
| H | Wrong-sector donor [20,22) | Yes | 42 | Exploratory |

Total training runs: **10**. Evaluations at steps 0/32/96 (primary
endpoint 32). Consolidation note: the frozen three-job layout is
executed as gates job → training job, with the held-out stage
folded into the end of the training job *after* the frozen primary
decision is computed and recorded in-job (trained connector
checkpoints cannot leave the job except as expiring artifacts;
retraining in a second job would not be the same checkpoints).
Scientific settings unchanged; if the primary rule fails, the
held-out slice is never acquired.
