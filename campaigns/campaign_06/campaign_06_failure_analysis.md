# Campaign 6 — Failure Analysis and Final Conclusion

The capacity control answered Campaign 5's open question. With random
sectors (so donor content cannot confuse the result), widening the
bottleneck from 128 to 256 improved macro development loss from 8.0091 to
7.3232, and tripling training improved it to 7.0706 — but the unconstrained
Campaign 3 child sits at 4.9021–4.0137 territory (4.9024 at 32 steps,
4.0137 at 96). The mandatory delta pathway throws away too much of the
host's representation: the suffix receives only a ~1.4% relative donor
change of a bottlenecked signal, and no tested budget or width recovers
the lost information.

## Programme-level conclusion (Campaigns 1–6)

- Unconstrained bridges: child works, donor ignored, random beats pretrained.
- Mandatory bridges: donor used and (at one boundary) weakly beneficial,
  but the pathway itself is too lossy to be useful.
- Width, placement, sequence-awareness, statistical conditioning, and
  budget scaling were each tested with matched controls; none produced
  stable, useful pretrained inheritance.

This is a rigorous negative for the Mamba-130M/Qwen2.5-Math-1.5B pairing,
the tested interface families, 1–6-block sectors, and CPU-scale budgets.
It does not rule out other pairings, objectives that reward donor
information content directly, or training the interfaces at far larger
scale — but within this project's tested space, the defensible final
answer is: **structural surgery demonstrated; useful capability
inheritance not achieved.**
