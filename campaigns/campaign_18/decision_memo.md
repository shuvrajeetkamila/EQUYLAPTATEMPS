# Campaign 18 — Decision memo

**Label: `PLACEMENT_AWARE_GRAFT_DESIGN_READY`** (frozen precedence,
`decision_rules.md`). Campaign 19 is **not** authorized.

In plain terms: we now have a reliable map of how much it hurts this
model to move its own blocks around, and the map works on
placements it had never seen (its error was about half the naive
guess). Three lessons stand out: the first and last blocks are
irreplaceable in place; sending a block backwards toward the input
is much worse than sending it forwards; and how far a block moves
matters less than *which* block and *which way*. Distant section
swaps remain catastrophic, confirming Campaign 17 with a full map
rather than two points.

That earns a *design*, not a transplant. The Campaign 19 draft is
complete and honest about its controls, but running foreign weights
is a separate decision — yours — and would need its own frozen
preregistration. Nothing in Campaign 18 used a foreign model,
training, or the sealed capability held-out set.
