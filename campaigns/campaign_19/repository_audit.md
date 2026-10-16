# Campaign 19 — Repository audit (preparation stage)

## Live state

- Session-restore artifact recurred; recovered non-destructively to
  remote `5475c8f95fae7cb6f9c3b186bc25ca7d2d389703` (Campaign 18
  final); tree clean; PR #1 open and unmerged.
- Campaign 18 artifacts re-read: label
  `PLACEMENT_AWARE_GRAFT_DESIGN_READY`, gates G1–G5 pass, G6 not
  granted; map/predictor evidence in
  `campaigns/campaign_18/map_38907871417/` and
  `heldout_38911632615/` (SHA-verified bundles).
- The Campaign 18 draft (`campaign_19_pilot_preregistration_draft.md`)
  proposed: SmolLM recipient sector [11,13), Mamba donor, residual
  connectors, LR 3e-4, step-0/32/96 schedule. The frozen Campaign 19
  preregistration below refines it in these documented ways (draft
  itself is historical and unchanged):
  1. Connectors must bridge widths 576↔768, so the same-width
     residual seam cannot be used; the preregistered connector is
     the historical cross-width `DenseSeam` pair (documented change,
     forced by tensor geometry).
  2. Exact donor sector frozen as donor blocks [11,13) and the
     donor-validity gate (D0) made explicit.
  3. Conditions expanded to the full A–H control set with seeds
     {42, 123, 31415} and predeclared decision rules.
  4. LR 3e-4 verified against `campaigns/campaign_16/config/
     preregistration.json` (`baseline_lr: 0.0003`), not assumed.

## Feasibility (verified from repository evidence, not run here)

- Both checkpoints are pinned and were repeatedly loaded by prior
  Actions jobs (Mamba: Campaigns 1–13; SmolLM: Campaigns 7–18).
  Recipient: `HuggingFaceTB/SmolLM-135M@1d461723…` (30 blocks,
  width 576, tied embeddings, safetensors recorded in Campaign 7
  registry). Donor: `state-spaces/mamba-130m-hf@1e76775f…` (24
  blocks, width 768; safetensors SHA-256 `1a5ed29c…` recorded in
  Campaign 1 provenance). Apache-2.0 lineage documented.
- `StackSegment` already executes both block families with their
  native call conventions; `DenseSeam` bridges arbitrary widths;
  the Campaign 11 within-tensor scramble is architecture-generic.
- What is missing (built in this preparation): a foreign-graft
  builder on the Campaign 17/18 placement engine with manifests,
  donor-freezing/scramble/random-control machinery, and its offline
  tests. No real-weight foreign experiment has been run.
