# Campaign 14 — Repository Audit (Phase 0)

**Date:** 2026-10-14
**Repository:** `shuvrajeetkamila/EQUYLAPTATEMPS`, branch
`arena/97efdf4a-equylaptatemps`
**Campaign 14 starts from:** `fa1f8daaff186b3217ab2b6a21c2491dc157aad9`
("Campaign 13 complete: Outcome C"), verified as remote HEAD.

## Starting state (verified, not assumed)

- Fresh checkout presented the known session-restore artifact;
  recovered non-destructively to remote HEAD; `git status
  --porcelain` = 0 afterwards. No history rewritten, nothing
  force-pushed.
- PR #1: OPEN, unmerged. Latest CI: `38773938000` "Offline surgery
  checks" success.
- Campaign 13 evidence present and complete (21 runs + probe +
  validation bundles, all SHA-verified at import in Campaign 13).

## Campaign 13 verification (independent, from raw records)

Recomputed from `campaigns/campaign_13/screen32_38767422815/` and
`selftest_38770694519/` records (macro = mean of language/math dev
token NLL in each record's `validation.groups`):

- Native recipient (SmolLM-135M @ `1d461723…`): macro **2.7731**;
  native donor (SmolLM-135M-Instruct @ `fcc320f4…`): **3.0369**.
- Intact children (32 steps): 7.5618 / 7.3432 / 7.5167 (seeds
  42/123/31415); self-reconstruction child F: **7.4577**.
- Child structure (every record): prefix 11 + graft 6 + suffix
  17 = **34 blocks** (native: 30); `removed_recipient_blocks` = 2;
  trainable seam parameters 667,008; `positive_control: true`;
  provenance mode `same_family_positive_control`;
  `parent_weights_unchanged: true`; `test_metrics: null`.
- All published Campaign 13 numbers match the raw records; the
  reported ≈2.77 vs ≈7.3–7.6 gap is real and not a reporting
  artifact. The full causal decomposition is in
  `campaign_13_reanalysis.md`.

## Environment

Local sandbox: CPU-only, no Hugging Face access; `.venv`
(torch 2.5.1 CPU) rebuilt after the restore artifact. Real-weight
diagnostics run on GitHub Actions via the established commit-tag
workflows (Campaign 11 Amendment 01). Precision float32, 2
threads, eager attention — the campaigns' standard configuration.
