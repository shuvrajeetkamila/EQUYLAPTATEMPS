# Campaign 14 — Decision Memo

**Outcome: RECONSTRUCTION_VALIDATED.** The surgery pipeline is
proven capable of exact, function-preserving reconstruction, and
the Campaign 13 damage is fully localized:

| Factor | Initial damage (macro NLL) | Repairable? |
|---|---|---|
| Standard connector seams | +21.69 (alone, native placement) | **Yes — exactly**, by residual-identity initialization |
| C13 placement (relocate/remove/duplicate/deepen) | +9.24 (seams neutralized) | No — only avoidable or trainable |
| Historical combined start | +11.47 (14.25 abs.) | Training recovered to 7.46 (of which placement damage dominates the remainder) |

**Decisions:**
1. Adopt `residual_identity` seams as the default interface for
   any future campaign in which donor and recipient widths match,
   and require any new connector to pass an exact-reconstruction
   gate (this campaign's Gates 2–3) before use. Where widths
   differ, an exactly-identity connector is mathematically
   unavailable; that case needs its own preregistered treatment
   and must never borrow this campaign's validation.
2. Treat placement (which blocks are removed/duplicated/reordered)
   as a first-class experimental variable with its own damage
   budget, measured with seams neutralized — Campaign 13's design
   changed placement, depth, and connector at once.
3. Do not interpret short-budget trained scores as evidence about
   initialization quality: at 32 steps the historically damaged
   start trained further (7.46) than the identity start (8.46).
   Initial damage and training recovery are separate quantities
   and are now always reported separately.

**Single recommended next experiment (not started):** rerun the
Campaign 13 positive-control design (same placement, same budget,
three seeds) with residual-identity seams, preregistered, to test
whether the identity-preserving start improves the *self* control
at seed scale and clarifies the V7/V8 inversion — the earliest
unresolved problem on the path back to heterogeneous transfer.
Only if that control behaves sanely should a heterogeneous
donor campaign resume, using native-depth-respecting placement
where possible and the reconstruction gates as entry criteria.

**Not authorized by this memo:** merging PR #1, releases,
held-out evaluation, Campaign 15.
