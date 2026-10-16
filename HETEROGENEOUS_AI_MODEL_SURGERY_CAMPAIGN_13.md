# HETEROGENEOUS AI MODEL SURGERY — Campaign 13

## The Same-Family Positive Control: Can the surgery pipeline preserve learned structure when donor and recipient are genuinely compatible?

**Classification: Outcome C** — the positive control does not pass.
Even with a maximally compatible donor, the pipeline shows no
reliable learned-structure advantage. Promotion: no. Held-out:
`NOT_RUN_NO_PROMOTION`.

### The question

After Campaigns 11–12 found no reliable advantage for an intact
donor over scrambled copies of itself across the Mamba–SmolLM
family boundary, two explanations remained: the two families are
incompatible, or the surgery method itself cannot preserve learned
structure. Campaign 13 tested the second explanation directly:
transplant a sector from SmolLM-135M-Instruct into its own base
model, SmolLM-135M — same architecture, same tokenizer family,
same ancestry — through the identical pipeline, against the same
scrambled controls. An explicit, recorded `positive_control`
switch was added to the code for this (the pipeline normally
*refuses* related parents); the cross-family safeguards are
unchanged.

### The answer, in numbers (macro development NLL, lower is better)

Intact donor's advantage over each scramble, 32 training steps:

| Seed | vs within-row scramble | vs full scramble |
|---|---|---|
| 42 | +0.43% | +0.17% |
| 123 | +5.79% | +6.13% |
| 31415 | +1.89% | −1.14% |

One strong seed, one negligible seed, one reversal — the same
pattern as the cross-family campaigns, failing the frozen bar.
But the campaign's most revealing number is a reference point:
the untouched recipient scores **2.77**, while every surgical
child scores **7.3–7.6** — including a child given the recipient's
*own* sector back (7.46). The operation itself, at these training
budgets, costs far more quality than any donor arrangement could
plausibly repay. The method's bottleneck is not which weights go
in; it is that the transplant procedure badly damages the host
and 32–96 tiny training steps cannot heal it.

### What it means

The leading explanation for four campaigns of null results shifts:
not "the families are incompatible," but "this surgery regime —
frozen graft, small trained connectors, very short training —
cannot yet preserve learned structure even under ideal
conditions." Detecting donor knowledge will first require
reducing the damage of the operation itself.

### Record

Preregistration frozen before training (commit `6e6b41b`); jobs
`38766385142` (probe), `38767049795` (validation), `38767422815`,
`38770695036`, `38770694519` (self-test), all SHA-256-verified.
21/21 runs reported; 277 tests pass; lint clean. Full detail:
`campaigns/campaign_13/final_report.md` and `decision_memo.md`.
