# Campaign 14 — Surgery Damage Elimination (complete)

**Outcome: RECONSTRUCTION_VALIDATED.**

Question: why did the surgical pipeline damage even the recipient's
own sector (Campaign 13: 2.77 → 7.46)? Answer, measured factor by
factor at initialization (macro dev NLL, native 2.7731):

- Extraction/insertion machinery: **exact** — native-placement
  reconstruction is bit-identical (273/273 tensors, logit diff 0.0).
- Standard connector seams: **+21.69** damage alone (2.77 → 24.47).
- Campaign 13 placement (remove/duplicate/reorder/deepen):
  **+9.24** with seams neutralized.
- Historical combined start 14.25; 32 seam-training steps recovered
  only to 7.46 (replicated here as 7.4578).

Repair: a zero-initialized residual seam (`ResidualDenseSeam`,
opt-in `residual_identity` interface) is exactly identity at
initialization — the reconstructed child scores 2.7731, degradation
0.0000. Full record: `campaigns/campaign_14/final_report.md` and
`decision_memo.md`. Held-out: NOT_RUN_NO_PROMOTION.
