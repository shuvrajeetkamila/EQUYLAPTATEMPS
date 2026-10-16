# Campaign 18 — Architecture and mechanism analysis (Stage 6)

## What the recipient's block sequence is

SmolLM-135M is a 30-block Llama decoder: each block refines a
residual stream with RMSNorm, RoPE self-attention, and an MLP.
Position information enters through RoPE (global token positions,
unchanged by block reordering) — there are no per-block position
embeddings. Naively, one might hope blocks are interchangeable
"refinement steps." The Campaign 18 map shows they are not, and
the diagnostics below indicate why.

## Evidence (development set; run 38907871417)

1. **Boundary blocks are load-bearing.** Swapping blocks 0↔1 costs
   +6.36 NLL and swapping 28↔29 costs +10.59, while interior
   adjacent swaps cost +0.03…+0.33. The first block translates
   token embeddings into the stream's working representation and
   the last prepares it for the final norm/head; neither role is
   shared by interior blocks.
2. **Direction asymmetry.** Moving a block *earlier* (toward the
   input) is consistently far more damaging than moving it later by
   the same distance (block 8: −8 → +7.42 vs +8 → +0.22; block 14:
   −8 → +2.92 vs +8 → +0.44; block 26: −8 → +2.74). A block moved
   earlier forces every downstream block to consume a less-refined
   stream than it was trained on, and its own output then feeds
   blocks expecting a later-stage representation.
3. **Representation divergence is localized at graft boundaries.**
   Boundary hidden-state diagnostics (one frozen batch; cosine of
   each executed block's output vs the native block at the same
   position): the section exchange collapses cosine to 0.23 at the
   first exchanged position, recovering only partially (0.57 at the
   final position); the long single-block move keeps cosine ≥ 0.96
   everywhere yet still costs +0.44 — small, distributed drift
   rather than a single rupture. Similar norms would not have
   proved compatibility; here even high cosine coexists with real
   damage, confirming these diagnostics are descriptive, not
   exculpatory.

## Plain-English version

Think of the 30 blocks as 30 specialists on an assembly line, each
trained to receive the product in exactly the state the previous
specialist left it. Interior neighbors can trade places with mild
consequences; the first and last specialists cannot. Sending a
part *backwards* to an earlier station breaks the expectations of
every station after it; sending it forwards is gentler because
later stations receive something merely over- or under-processed,
not re-ordered. Nothing here proves other architectures behave
identically — this is a within-model finding for one 30-block
transformer, stated as such.

## Limitations

One checkpoint, one evaluation panel, deterministic scoring;
mechanism diagnostics use a single batch and three conditions.
Causal mechanism claims beyond the directional/boundary patterns
would require activation-level interventions not preregistered here.
