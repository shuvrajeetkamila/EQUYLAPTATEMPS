# Campaign 13 — Decision Memo

**Date:** 2026-10-14
**Decision:** Classify Campaign 13 as **Outcome C** (positive
control does not pass). Do not promote. Held-out remains
`NOT_RUN_NO_PROMOTION`. Do not begin a further campaign
automatically.

## Why

The campaign gave the learned-structure hypothesis its friendliest
possible test: identical architecture, tokenizer family, and
lineage; a real, quantified donor-recipient weight difference
(5.64% sector RMS); bit-exact scramble controls; healthy graft
activity; causal ablation support. The intact sector still failed
the frozen per-seed gate (seed-42 margins of 0.43%/0.17% against a
2% bar; a reversal against the full scramble at seed 31415). The
96-step panel converges to indistinguishability at seed 42.

The decisive context is the integration damage: the surgical
child scores ≈7.3–7.6 macro NLL where the untouched recipient
scores 2.77 — and even grafting the recipient's *own* sector back
scores 7.46. A pipeline that cannot yet preserve an (almost)
identity transplant cannot be expected to reveal a 5.6%-scale
learned difference, and did not.

## Consequences for the programme

- The Campaign 9–12 cross-family nulls should no longer be read
  primarily as "the families are incompatible." Campaign 13
  shows the same absence of reliable arrangement sensitivity
  *within* a family, under a regime whose dominant effect is
  surgery-induced damage. The bottleneck is the surgery regime
  (frozen graft + small trained seams + very short training), at
  least at these budgets.
- The positive-control methodology worked as intended: it was
  capable of returning a clear negative with valid controls, and
  it localized the problem. That is the campaign's contribution.
- Recommended next direction (not authorized): reduce integration
  damage before re-testing transfer — identity-preserving seam
  initialization and/or grafting at native depth, with the
  self-reconstruction child (F) as the yardstick (its approach
  toward native-recipient quality bounds what any donor could
  achieve). Any continuation requires a new preregistration with
  the same scramble controls and per-seed gates.

## Compliance notes

Preregistration frozen (`6e6b41b`) before confirmatory training;
the same-family opt-in is explicit, defaulted-off, and recorded
in every run record (`positive_control: true`); cross-family
protections and historical artifacts unchanged (golden archive
still validates); 21/21 runs reported; no amendments; held-out
sealed by frozen policy for all outcomes.
