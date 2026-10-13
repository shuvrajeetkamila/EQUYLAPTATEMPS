# Campaign 5 — Failure Analysis

The width/placement hypothesis is falsified at this scale. Making the donor
bigger makes it *transform more* (relative update 0.05→0.18 from 1 to 6 blocks)
without making the child *better*: macro loss stays ~7.8–8.1 for every sector,
language stays ~1.8x the host, and pretrained-vs-random differences scatter
around zero (+4.5% to −2.1% for the same sector across seeds). Position
(early/middle/late) is likewise irrelevant. Combined with Campaign 4, the
picture is consistent: the mandatory-use architecture guarantees donor
*dependence*, but the donor's pretrained computation at this Mamba boundary —
at any tested width or location — carries no stable, useful advantage over
random blocks, and the forced pathway itself costs too much performance.
Campaign 4's isolated two-seed w2 benefit should now be read as the favorable
tail of a noisy distribution, not a scalable effect. This is a defensible
negative for this parent pairing, interface family (bottleneck-128 + delta),
sector ranges (1–6 blocks, three positions), and CPU budgets (32 steps);
it does not generalize to all heterogeneous pairings.
