# Campaign 4 — Failure Analysis

## What changed and what it proved

Campaign 3's child ignored its donor because the bridges could pass a
host-friendly signal straight through a near-identity sector. Campaign 4
removed that freedom in two ways: (1) the **donor-delta pathway** — the
suffix receives only what the donor *changed*; (2) **128-wide bottleneck
interfaces** capping seam-to-seam capacity.

The forcing worked mechanically. Under delta designs, removing the donor
now costs 17–49% macro development NLL (Campaign 3: <0.2%), and zeroing it
is catastrophic. The child genuinely routes through the donor.

## The remaining failure

Forcing use is not the same as inheriting capability:

- All forced children are much worse than the unconstrained Campaign 3
  child (macro 7.70–8.02 vs 5.54) and fail the language-retention gate
  (1.82–1.85× host). The donor's transformation, when it is the only
  signal available, is a poor carrier for the host's knowledge — the
  bridges at a 128-wide bottleneck plus a ~8% donor update cannot
  reconstruct what the host prefix knew.
- The pretrained donor's edge over random blocks is real but tiny: only
  the bottleneck+delta candidate beat its matched twin at both seeds
  (+2.8%, +1.7%). In plain terms, the transplanted chapters are now being
  *read* — the book gets slightly worse with blank chapters — but what is
  read does not yet make the book good.

## Classification reasoning

This is **Outcome B (partial evidence)**, not A (Gate D fails decisively;
no promotion) and not C (unlike Campaign 3, a preregistered
pretrained-specific benefit with consistent direction across two seeds and
strong donor dependence now exists). The benefit's small magnitude, the
ablation distribution-shift caveat (bypass/zero feed the output bridge
unfamiliar signals), and the destroyed absolute performance are recorded
as limitations, not hidden.

## What the evidence rules out and leaves open

Rules out: "the bridges merely needed sequence context" (C3) and "the
child merely needed to be forced" (C4) as sufficient explanations.
Leaves open: whether donor blocks selected/placed differently, wider
donor sectors, or training objectives that reward donor *information*
(not just donor *use*) can convert this partial contribution into useful
capability. The single best-supported next hypothesis is that the donor
sector's usable signal is too weak at this boundary (2 of 28 blocks,
~8% relative update), so a preregistered donor-placement/sector-width
study under the mandatory-use architecture is the justified next step —
not more interface complexity at the same boundary.
