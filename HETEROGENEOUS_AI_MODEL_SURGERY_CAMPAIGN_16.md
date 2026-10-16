# Campaign 16 — Optimization-Dynamics Investigation (complete)

**Outcome: LEARNING_RATE_SENSITIVE** (Gate E mechanism supported).

Why did the damaged connector beat the healthy one at 32 steps
(Campaign 15)? An 18-run sweep (2 connectors x LR {9e-5, 3e-4,
9e-4} x 3 seeds) answers: the paired difference at step 32 moves
monotonically with learning rate (-2.59 -> -0.99 -> +0.17), and at
step 96 from -1.04 -> +0.16 -> +0.91. The pass-through connector
has ~0.77x the early gradient norms, starts from exactly zero,
and is simply under-trained at the shared learning rate; given
more LR or more steps it overtakes, producing the best children
measured (6.04). Its remaining deficit vs the native 2.77 is
placement damage (+9.24), which no connector fixes. Full record:
campaigns/campaign_16/final_report.md. Held-out sealed.
