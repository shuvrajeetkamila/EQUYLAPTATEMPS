# Campaign 14 — Independent Reanalysis of Campaign 13's Damage

**Question:** why did Campaign 13's surgical children score
≈7.3–7.6 macro dev NLL when the untouched recipient scores
2.7731 — including the child that received the recipient's own
sector back (7.4577)?

**Method:** source-code inspection of the surgery path
(`model.py`, `backbones.py`, `alignment.py`, Campaign 13 runner)
plus recomputation from Campaign 13's raw records. No new runs
were performed for this reanalysis; the quantitative decomposition
is Campaign 14's diagnostic matrix (see `preregistration.md`).

## Findings, mapped to the brief's candidate causes

**1. Genuinely altered model function — YES, by construction.**
The child is not the recipient's computation with a part swapped;
it is a different network. Four independent alterations compound:

**2. Incorrect layer/tensor mapping — NO evidence of a bug.**
Graft blocks are the genuine parent blocks (per-tensor SHA-256
records in Campaigns 11–13 match the parents; parameter counts
are consistent everywhere). The mapping does exactly what the
design specifies; the design itself is the problem, viz.:

**5. Relocation / re-depth / removal / duplication — YES, major.**
Native block order is 0…29. The Campaign 13 child executes
0–10, then donor **24–29**, then 13–29: blocks 11–12 are removed,
blocks 24–29 run twice (once as graft, once inside the suffix),
late blocks feed early blocks, and depth grows 30 → 34. Even the
"self" child (F) therefore computes a very different function
from the recipient. Campaign 13's report noted relocation; the
duplication and reordering deserve equal emphasis.

**4. Connector dimensions/initialization — YES, major.**
Two `DenseSeam`s (LayerNorm → Linear) sit in the residual stream.
At initialization the input seam is an orthogonal projection
recolored by covariance statistics, the output seam a generic
Xavier projection (Campaign 10's `natgeom_in_trained_out`). A
LayerNorm in the residual stream already destroys the stream's
scale/offset information that Llama blocks expect (each block
applies its own input norm); a non-identity projection then
rotates the representation. The seams are *never* identity, in
any campaign to date. Which share of the damage is seams vs
placement is measurable and is Campaign 14's Stage C/D matrix.

**3. Changed residual/normalization path — YES (see 4).**
No other normalization substitution occurs; block-internal norms
are the parents' own, frozen.

**6. Training / checkpoint behavior — contributing.**
Only the seams train (667,008 parameters), 32 steps on 64 tiny
training examples. Campaign 13's F child recovered to 7.4577 from
an untrained value Campaign 14 will measure (expected worse);
recovery is partial by budget, not by checkpoint selection (the
runner always evaluates the final step; no best-checkpoint
selection exists to suspect).

**7. Evaluation/tokenization/scoring inconsistency — NO.**
Within the same Campaign 13 jobs, the same encoding and scoring
code produced native recipient 2.7731 and child ≈7.5; the child
uses the recipient's own tokenizer; checkpoint revisions and
tokenizer fingerprints are recorded and consistent; the native
score reproduces across the screen and self-test jobs exactly
(2.7731 in both). The gap is in the model function, not the
measurement.

**8. Interaction.** The factors interact: mis-placed blocks
receive seam-distorted inputs in the wrong order, and seam-only
training must compensate for all of it at once. Campaign 14
therefore isolates factors in stages rather than assuming
additivity.

## Conclusion carried into Campaign 14

The damage is overdetermined by design features (placement +
duplication + non-identity seams + tiny seam-only training), not
by a coding slip. The repair strategy follows: (i) prove the
extraction/insertion machinery exact at native placement with
seams neutralized; (ii) make the connector function-preserving at
initialization (zero-initialized residual seam: exact identity,
trainable); (iii) quantify the placement damage with seams
neutralized; (iv) only then compare trained repair vs the
historical configuration.
