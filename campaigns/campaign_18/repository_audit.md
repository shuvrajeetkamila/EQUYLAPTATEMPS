# Campaign 18 — Repository audit (Stage 0)

## Live state (verified at campaign start)

- Session-restore artifact recurred (local `ecc8b0d`, dirty tree);
  recovered non-destructively to remote
  `dc386bc96cd14c81c408e5f121c4dc386083b650`; tree clean.
- PR #1: open, unmerged ("Campaigns 1-17 complete — native-depth
  placement exact; structural exchange damage remains dominant").

## Campaign 17 claims independently re-verified from raw evidence

Read directly from `campaigns/campaign_17/diag_38899506528/diagnostics.json`
(SHA-verified bundle of Actions run 38899506528, commit 5a8ebf4):

| Claim | Verification |
|---|---|
| Native macro 2.7731 | **Confirmed**: 2.773126, repeated exactly (repeat Δ 0.0) |
| B historical reconstruction exact | **Confirmed**: macro 2.773126, logit diffs 0.0, tensors 270/270 |
| C identity replacement exact | **Confirmed**: macro 2.773126, logit diffs 0.0, tensors 270/270 |
| D1 exchange [11,13)↔[24,26) | **Confirmed**: 11.168408 (+8.3953), moved blocks exactly the four declared |
| D2 swap 15↔16 | **Confirmed**: 2.927872 (+0.1547) |
| Depth/identity preservation | **Confirmed**: manifests show 30→30, removed [], duplicated [] |
| Parents unchanged | **Confirmed**: `parents_unchanged: true` in raw record |

**Discrepancies:** none found. Campaign 17 numbers may be used as
Campaign 18 anchors (A/B/C/D1/D2 are also replicated inside the
Campaign 18 design, so no conclusion rests on trust alone).

## Baseline definition (unchanged from Campaigns 14–17)

- Recipient: `HuggingFaceTB/SmolLM-135M@1d461723…`, 30 Llama blocks,
  width 576, tied embeddings; loaded in float32 on CPU.
- Metric: macro dev NLL = mean of mean token NLL over the frozen dev
  panel (16 GSM8K-style math + 16 WikiText-2 text examples, 192-char
  chunks, the Campaign 14–17 panel). Lower is better. It is a mean
  negative log-likelihood (cross-entropy), **not** an accuracy.
  Machine-readable anchor: `baseline_reference.json`.
- Engine: `src/equylaptatemps/native_placement.py` (Campaign 17),
  permutation-only execution orders with structural manifests.
