# Campaign 6 — Preregistration (frozen before training)

## Question

Can a mandatory-use child (bottleneck interfaces + donor-delta) with a
**random** sector approach the performance of the unconstrained Campaign 3
child with a random sector, at matched budget? If not, the forced pathway
itself — not donor content — is the binding limitation.

## Verified references (same development protocol, random sectors)

- Unconstrained full-width attention seams, 32 steps: macro 4.9024 (C3).
- Unconstrained full-width attention seams, 96 steps: macro 4.0137 (C3).
- Mandatory bottleneck-128 + delta, 32 steps: macro 8.0091 (C4 twin).

## New runs (random sectors only, donor [20,22), seed 42, host partition [11,13))

1. `c06_bn256_delta_rand_42` — bottleneck widened 128 → 256, 32 steps.
2. `c06_bn128_delta_rand_96` — bottleneck 128, budget tripled to 96 steps.

## Frozen decision rule

Capacity is **remediable** if either run reaches macro <= 1.10 x its
matched-budget unconstrained reference (32-step: 5.393; 96-step: 4.415).
If both fail, the mandatory pathway is intrinsically lossy at these
budgets and the defensible conclusion is a final negative for this
pairing/interface family at this scale. This study uses random sectors
only: it cannot promote any candidate and authorizes no held-out run.
