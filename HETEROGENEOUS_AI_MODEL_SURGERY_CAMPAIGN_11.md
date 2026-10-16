# HETEROGENEOUS AI MODEL SURGERY — Campaign 11

## The Learned-Structure Test: Pretrained Donor vs Statistics-Preserved, Structure-Destroyed Controls

**Classification: Outcome C** — the control was valid, and the intact
donor sector showed **no reliable advantage** over a scrambled copy
of itself. Promotion: no. Held-out: `NOT_RUN_NO_PROMOTION`.

### The question

Campaigns 9–10 found a small, fragile advantage for a pretrained
donor sector over random twins. Campaign 11 asked what that
advantage is made of: the donor's *learned internal arrangement*, or
merely the *statistics and scale* of its weights? The test scrambles
each of the donor sector's 42 weight matrices internally (one frozen
shuffle seed): every value survives, every per-tensor statistic is
identical, but the learned arrangement is destroyed. The scrambled
sector then gets exactly the same treatment the intact sector got in
Campaign 10.

### The answer, in numbers (macro development NLL, lower is better)

| Steps | Seed | Intact donor | Scrambled donor | Intact advantage |
|---|---|---|---|---|
| 32 | 42 | 8.3665 | 8.6808 | +3.62% |
| 32 | 123 | 8.3132 | 8.3098 | −0.04% |
| 96 | 42 | 7.8392 | 7.7141 | −1.62% |

The scrambled sector was a genuine, active control (the child used
it as much as the intact one; all frozen validity checks passed).
Yet the intact sector's edge appeared at one seed, vanished at the
other, and reversed with more training. Under the rules frozen
before any run, that is **Outcome C**: the earlier donor-specific
effects are best explained by weight statistics and scale passing
through a geometry-aware interface — not by learned structure that
transfers between model families. Absolute quality also remains far
from useful (language ≈1.8× the host vs a 1.50× ceiling).

### Record

Preregistration frozen before training (commit `fb90ebb`); two
dated amendments cover only the execution trigger and the budget-run
authorization. Evidence: GitHub Actions jobs `38713676113` and
`38715108842`, SHA-256-verified on import. 258 tests pass; lint
clean. Full detail: `campaigns/campaign_11/final_report.md` and
`decision_memo.md`.
