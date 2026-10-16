# Campaign 19 — Preregistration: First Controlled Foreign-Graft Pilot

Status: **PREPARED, NOT AUTHORIZED.** No foreign-graft training or
capability experiment may run until the user explicitly authorizes
this frozen document in a later instruction.

## Question

Does a frozen foreign pretrained component provide a reproducible,
donor-specific benefit inside the recipient, beyond the same
architecture and trainable connector capacity without the intact
donor's learned structure?

## Frozen design

- Recipient: `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`
  (frozen; 30 blocks, width 576, tied embeddings).
- Donor: `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
  (frozen; 24 blocks, width 768).
- Graft: recipient blocks [11,13) replaced at native depth by donor
  blocks [11,13) via `build_foreign_graft_child` (Campaign 19
  preparation). All other recipient blocks keep identity and order;
  depth stays 30; replaced blocks are recorded in the manifest.
- Connectors: seeded `DenseSeam` bridges 576→768 and 768→576
  (seed = run seed). Only connector parameters train. (The Campaign
  18 draft's "residual connectors" cannot bridge unequal widths;
  this documented substitution is forced by tensor geometry.)
- Training (post-authorization only): Campaign 15/16 segment-training
  procedure; LR **3e-4** (verified `baseline_lr` in
  `campaigns/campaign_16/config/preregistration.json`); 96 steps;
  evaluations at steps 0/32/96; seeds {42, 123, 31415}; primary
  endpoint step 32 (Campaign 15/16 convention). No LR search.
- Metric: macro dev NLL on the frozen Campaign 14–18 panel (native
  reference 2.773126). Damage = condition − native.

## Gates and conditions

- **D0 donor validity** (donor alone, own tokenizer): intact donor
  text-panel NLL finite and ≥ 0.05 better than the same donor with
  sector [11,13) scrambled in-body (threshold: documented
  engineering rationale, exploratory). Failure ⇒ pilot blocked.
- **B identity control**: recipient's own [11,13) through the same
  pathway must be exact (frozen tolerances); failure blocks C–H.
- **A** native. **C** intact foreign graft. **D** scrambled donor
  (Campaign 11 within-tensor permutation, seed = run seed; preserves
  per-tensor value multisets, destroys arrangement). **E**
  distribution-matched random donor sector (same shapes/counts;
  learned structure absent). **F** ablation: trained C with
  `bypass_graft` (removes donor+connectors together; limitation
  recorded). **G** restoration: ablation off, must reproduce C
  exactly. **H** (exploratory, seed 42): donor blocks [20,22).
- C/D/E receive identical connectors, budgets, and evaluation.

## Decision rules (frozen; see decision_rules.md)

Primary success requires ALL of: B exact; mean intact-vs-best-control
advantage ≥ **0.25 NLL** at step 32 (Campaign 16 ranking threshold)
with ≥ 2/3 seeds positive; ablation removes ≥ 50% of the advantage;
preservation damage of trained C ≤ +4.50 (rationale: not worse than
the best historical trained child, +4.0, by more than 0.5). Any seed
sign reversal is reported, not averaged away.

## Held-out protection

Historical capability held-out sets are never touched. Campaign 19
held-out = a new disjoint slice (procedure frozen in
`dataset_registry.json`; split hashes recorded at first authorized
acquisition, before training, and logged). It opens only if the
primary success rule is met; otherwise `NOT_RUN_NO_PROMOTION`.

## Not authorized by this document

Any training run, foreign-condition scoring beyond D0/B gates,
held-out access, or design change. Changes require a dated amendment
and, where they alter the question or rules, renewed authorization.
