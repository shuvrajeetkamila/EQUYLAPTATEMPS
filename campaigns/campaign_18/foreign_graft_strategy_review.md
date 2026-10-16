# Campaign 18 — Foreign-graft strategy review (Stage 7, design only)

Campaign 18 ran **no** foreign condition. This review uses the
verified damage map to judge four placement strategies on paper and
by structural check. Two questions are kept separate throughout:
(1) *structural* insertability without deleting/duplicating/shifting
recipient blocks — answered by the engine and Campaign 17/18
controls; (2) *functional* compatibility of foreign weights — not
answered by any structural evidence.

## What the map implies for any strategy

- Recipient blocks must stay at their native depths: distant native
  movement costs up to +10.6 NLL. Any strategy that shifts recipient
  blocks to "make room" inherits that damage before the foreign
  component does anything.
- Boundary depths (first/last blocks) are the worst locations.
- Interior locations with mild *native* movement damage are
  candidates for investigation, but low movement damage does **not**
  imply foreign-weight compatibility (Stage 6 keeps these separate).

## Strategy A — Exact-location replacement

Replace one named recipient block/sector at its own depth; all other
blocks keep identity and order. Structural cost: zero by
construction (validated, Campaign 17 Mode C pattern with a foreign
sector substituted for the self sector). Requirements: matching
hidden width and block interface, or a declared connector. Failure
mode: foreign block receives/produces incompatible representations
(Campaign 13/14-scale damage is the honest prior). **Verdict: the
only strategy the map supports as a pilot skeleton.**

## Strategy B — Native-depth sector substitution

As A, for a multi-block sector. Same requirements plus sector-length
matching. The map shows multi-block native exchanges are the most
damaging native operations (+3.2…+8.4), so sector-level foreign
substitution starts from the harshest structural prior.
**Verdict: defer until a single-block/sector pilot justifies it.**

## Strategy C — Boundary-aware connector

Insert trained residual connectors at the graft boundaries (the
Campaign 14–16 interface family) while recipient blocks stay frozen
at native depths and the foreign block stays frozen. Requirements:
donor-statistics-free calibration rules frozen in advance (Campaign
16 showed strong LR sensitivity; any pilot must preregister its LR
from that evidence, not search). Failure modes: connector becomes
the real model (guarded by ablation + frozen-donor controls), LR
mis-set (Campaign 16). **Verdict: necessary companion to A for the
historical donor (below); not a standalone strategy.**

## Strategy D — Restricted local integration

Choose graft location from the map's low-damage interior regions.
**Verdict: a location-selection heuristic layered on A/C, not an
independent strategy; "low movement damage" must never be quoted as
"foreign-friendly" without the pilot's own controls.**

## Donor selection (repository-grounded, not convenience)

The project's established heterogeneous pair is the recipient
SmolLM-135M (width 576) with `state-spaces/mamba-130m-hf@1e76775…`
(width 768, 24 blocks) from Campaigns 1–12. Widths and block
contracts differ, so no unmodified foreign block can occupy a
recipient slot: Strategy A alone is structurally inapplicable, and
A+C (exact location + frozen declared connectors) is the honest
pilot form — exactly the Campaign 19 draft in
`campaign_19_pilot_preregistration_draft.md`. Same-family
substitutes would be positive controls, not transfer evidence.
