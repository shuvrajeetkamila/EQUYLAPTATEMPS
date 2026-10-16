# Campaign 15 — Damage and Training Analysis

**Outcome (mechanical, frozen rules): TRAINING_DEPENDENT.**

## The three quantities, kept separate

1. **Connector damage at installation** (Gate B): installing the
   old connector at the changed placement costs an additional
   **+1.25 to +2.29 NLL** over the identity connector at step 0
   (same topology, same weights elsewhere). The repaired
   connector adds zero connector damage, confirming Campaign 14
   at three seeds.
2. **Placement damage:** with the connector neutralized, the
   Campaign 13 placement itself costs +9.24 (12.0150 vs native
   2.7731; Campaign 14 decomposition, re-verified by the
   regression gate and by every new-condition run's step-0 =
   12.0150). No connector choice can remove this; both conditions
   pay it.
3. **Optimization after initialization:** the trajectories cross
   twice. The old connector converts its worse start into a lead
   by step 8 and holds it through step 64; the new connector
   improves more slowly but steadily and leads at step 96 at all
   three seeds.

## Why the early phase favors the old connector (interpretation)

The old seams begin with large, already-shaped projections: the
graft pathway immediately transmits a strong (if distorting)
signal, giving the 32-step optimizer something substantial to
correct. The residual seams begin transmitting *the unmodified
recipient stream* through the graft, but their learnable branch
starts at exactly zero and must grow — a slower early trajectory
under a fixed small learning rate, after which the cleaner
starting geometry overtakes. Post-training graft-activity records
are consistent with the new connector's seams ending as large
transforms (relative graft update ≈7.6 vs ≈0.3–1.1 for the old).
This is a dynamics interpretation of observed trajectories, not
a separately proven mechanism.

## What this means

- Campaign 14's repair is **real but not a free lunch**: it
  eliminates connector-inflicted damage (proven again), yet at
  the historically standard 32-step budget the damaged start
  trains to a better score — at every seed.
- **Budget choice changes the winner.** Any future comparison
  that reports only one budget (especially 32 steps) would be
  hostage to this crossover; trajectories, not endpoints, are
  the honest unit of comparison for connector questions.
- Neither connector rescues the placement: after 96 steps both
  children sit near 6.8–7.0 against a 2.77 native score. If the
  project wants transfer evidence, placement strategy — not
  connector tuning — is now the largest known lever.
