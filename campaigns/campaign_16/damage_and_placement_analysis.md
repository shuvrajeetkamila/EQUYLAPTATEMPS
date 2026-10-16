# Campaign 16 — Damage and Placement Analysis

Reference points (all from SHA-verified records): healthy native
recipient **2.7731**; changed placement with neutralized
connectors **12.0150** (placement damage **+9.24**); old connector
adds a further **+1.25…+2.29** at installation (seed-dependent);
new connector adds **0**.

After 96 steps of training, the best cell of the 18-run matrix
(new connector, LR 9e-4) reaches **6.04–6.11** — still ≈3.3 NLL
worse than the untouched recipient. No connector, learning rate,
or budget in this campaign recovers the placement cost; training
recovers at most about two-thirds of the combined initial damage
(14.25 → 6.04 in the best historical-configuration comparison).

Conclusion: the altered placement (blocks 11–12 removed, 24–29
duplicated, depth 30→34) is a structurally damaging topology
whose cost dominates every connector effect measured in
Campaigns 14–16. Connector science on top of it has diminishing
returns; the placement itself is the next problem to solve.
