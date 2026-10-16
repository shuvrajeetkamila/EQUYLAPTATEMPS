# Campaign 19 pilot — preregistration DRAFT (not authorized, not run)

Status: complete design awaiting a separate user authorization and
its own frozen preregistration commit. Campaign 18 does not execute
any part of this pilot.

## Pairing and placement (frozen candidates)

- Recipient: `HuggingFaceTB/SmolLM-135M@1d461723…` (frozen).
- Donor: `state-spaces/mamba-130m-hf@1e76775…` (frozen; Apache-2.0
  lineage documented in Campaign 7 records).
- Replacement: recipient sector [11,13) at its native depth replaced
  by a two-block donor sector, recipient otherwise untouched (engine
  guarantees: no deletion, duplication, or shifting).
- Interfaces: residual-identity connectors (Campaign 14 family) at
  both boundaries; only connector parameters train, at the Campaign
  16 baseline LR 3e-4 with the frozen step-0/32/96 schedule — no LR
  search (Campaign 16 sensitivity is the reason).

## Conditions (all on the frozen dev panel; held-out sealed)

1. Untouched recipient. 2. Recipient-own-sector identity replacement
   (must be exact; else stop). 3. Scrambled-donor control (donor
   block internals permuted under a frozen scheme). 4. Foreign graft,
   untrained connectors at identity. 5. Foreign graft, trained
   connectors. 6. Graft-ablated (bypass) versions of 4–5.

## Success evidence required

A learned-structure advantage means: trained foreign graft beats the
scrambled control and the untrained graft by a preregistered margin
on dev, replicates across 3 seeds, and only then opens held-out
under the standing promotion rule. Forward-pass success, loss
decrease, or beating the damaged historical child proves nothing.
