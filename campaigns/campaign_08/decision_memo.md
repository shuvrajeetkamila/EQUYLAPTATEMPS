# Campaign 8 Decision Memo — the single best next experiment

**Do not start Campaign 9 automatically.** This memo names the most
informative next step and why.

## What Campaign 8 changed about what we know

Calibration was the programme's best remaining "the interface is the problem"
hypothesis, and the 2×2 design killed it cleanly: host-statistics calibration
is a *general* conditioner that preferentially helps near-identity random
sectors. Combined with Campaign 6 (mandatory pathways are intrinsically
lossy), Campaign 5 (placement/width don't matter), and Campaign 7 (donor
family doesn't matter), the evidence now converges: **at these budgets, a
pretrained transformer's internal geometry is actively hard for a foreign
host to exploit through trained pointwise/linear interfaces, while random
blocks are easy because they barely transform anything (relative update
≈0.2–0.5 vs ≈19–58).** The donor's transformation magnitude — the very thing
that makes it "capable" natively — is what the interface cannot domesticate.

## Best next hypothesis (one)

**The unit of transfer is wrong, not the interface:** transplant the donor
sector *together with its native input embedding geometry* — i.e., let the
child feed the sector through the donor's own preceding block's output
distribution by including one donor block's *input-side RMSNorm statistics*
as a fixed (non-trained) donor-native preconditioner, rather than asking
trained seams to invent the donor's geometry from token loss alone. If even a
donor-native preconditioner cannot make the pretrained sector beat its twin,
the programme will have strong evidence that block-level cross-architecture
transplantation at this scale is fundamentally limited, and effort should
move to a different transfer unit (e.g., whole-prefix transplantation) or
stop. This is a recommendation for the owner's decision; nothing is launched.
