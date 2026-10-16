# HETEROGENEOUS AI MODEL SURGERY — Campaign 3 Final Report

**Sequence-Aware Semantic Alignment and Functional Capability Inheritance**
Date: 2026-10-13 · Branch `arena/97efdf4a-equylaptatemps` · PR #1

## Final decision: Outcome C — structural surgery without useful inheritance

The pretrained Qwen blocks are verifiably present, intact, frozen, and
executed inside the Mamba-host child, and the new sequence-aware bridges are
a genuine improvement over Campaign 2. But the donor sector still does not
contribute useful computation: removing it costs nothing, and replacing it
with randomly initialized blocks makes the child *better* at both tested
training budgets. That is the opposite of capability inheritance.

## 1. What was verified at the start

Repository, branch, PR #1, and Campaign 2 head `5ac671a` all matched the
reported state exactly; offline CI (Python 3.10/3.11) passed; both published
bridge-only checkpoints and the synthetic baseline were byte-identical to
their recorded hashes; 211 pre-existing tests passed. Full details:
`campaigns/campaign_03/campaign_03_initial_audit.md` and
`campaign_03_initial_state.json`.

## 2. What caused the original transplant to fail

Campaign 3 sharpened Campaign 2's diagnosis with three measurements:

- The famous scale gap (native donor input RMS 29.6 vs hybrid 0.82) is
  mostly a first-token spike (steady-state ≈4.3), so a single scalar gain
  cannot fix it — but fixing scale never created semantic alignment anyway.
- Inside the trained child, the donor sector changes its input by a relative
  norm of only ≈0.09: it is a near-identity passenger.
- The seam-to-seam path (input bridge → output bridge) carries everything
  the child needs. The bridges learn to write representations that pass
  through the foreign sector almost untouched.

## 3. Which sequence-aware interfaces were implemented

Three causal, residual, zero-initialized mixers placed after the unchanged
full dense projections (so each starts out numerically identical to the
Campaign 2 dense bridge): depthwise causal convolution (kernel 5),
bottleneck causal attention (128-wide, 4 heads), and bottleneck GRU (128).
Trainable parameters: 2,380,032 / 3,189,248 / 3,157,536 vs the dense
control's 2,366,208 (child total 217.5M; parents frozen). Causality is
unit-tested (future tokens cannot affect earlier outputs). No LoRA,
distillation, teacher targets, hidden-activation imitation, weight
averaging, ensembles, or routers were used.

## 4. Which experiments actually ran

Three opted-in connected CPU jobs, all completed successfully, sequential
and resumable, development data only:

- Screen (run `38367618192`, 22 min): dense-covariance control + the three
  sequence candidates, seed 42, 32 training steps each, with donor
  ablations and fresh-load reconstruction checks.
- Diagnostic confirm (run `38371104410`, 32 min): best candidate at 96
  steps, at seed 123, and as an identically trained random-sector twin.
- Diagnostic confirm 2 (run `38375777144`, 21 min): the random-sector twin
  at 96 steps, completing the budget-by-content matrix.

## 5. Which experiments were interrupted or blocked

None. Held-out language/mathematics confirmation was **not run by design**:
the frozen preregistration forbids it without promotion, and no candidate
was promoted. GPU work remains NOT_RUN (no GPU available). Candidate
binaries remain in the named GitHub Actions artifacts; local download of
those archives was not performed (recorded in the integrity report).

## 6. The best development result

Development macro token-NLL (lower is better; 16 language + 16 math
examples, host tokenizer):

| Candidate | Language | Math | Macro |
|---|---|---|---|
| Dense-covariance control | 6.6517 | 4.8275 | 5.7396 |
| Causal convolution | 6.5822 | 4.7110 | 5.6466 |
| **Causal attention** | **6.4577** | **4.6240** | **5.5409** |
| Bottleneck GRU | 6.4883 | 4.6062 | 5.5472 |
| Host (Mamba, native) | 4.5595 | 2.4818 | — |

All three sequence candidates passed the historical language-retention gate
(1.416–1.444× host vs the 1.50 ceiling) — the first bridges ever to do so —
but failed the frozen requirements to beat the dense control by 5% on
language, improve math by 10%, and show donor usage. The 96-step attention
child reached macro 4.3551, but identically so with the donor removed.

## 7. Held-out language and mathematics results

Not run (see §5). Campaign 2's held-out numbers stand unchanged: child
3.2796 bits/byte vs Mamba 1.7921 and Qwen 2.3431; math diagnostic Qwen
12/16, child 0/16.

## 8. Results against matched random-sector controls

| Budget | Pretrained donor | Random-sector twin |
|---|---|---|
| 32 steps | 5.5409 | **4.9024** |
| 96 steps | 4.3551 | **4.0137** |

Random blocks beat the real pretrained blocks at both budgets under
identical interfaces, data, seeds, and training.

## 9. Whether the donor makes a measurable useful contribution

No. Removing the donor (`bypass_graft`) changes macro NLL by <0.2% in every
run; its relative transformation of the signal is ≈9% (random twin ≈2%);
zeroing its output is catastrophic only because the output bridge then
receives silence, not because donor computation is lost.

## 10. Frozen-weight and checkpoint-integrity results

Parent fingerprints unchanged in all runs; every non-random candidate's
bridge-only checkpoint reconstructed with identical logits; evidence
bundles SHA-256-verified on import. Details:
`campaigns/campaign_03/campaign_03_integrity_report.md`.

## 11. Tests and CI

223 offline tests pass locally (221 pre-existing + new sequence-seam and
Campaign 3 traceability tests); lint and format clean. Connected CI status
is reported in the PR.

## 12. Latest GitHub commit and PR status

See PR #1 (updated with this report); the final commit hash is recorded in
the PR summary at publication time.

## 13. Final scientific outcome

**Outcome C — Structural surgery without useful inheritance**, now supported
by unusually direct controls: the transplant is real, the execution is
real, the bridges genuinely improved — and the donor's pretrained knowledge
is still not being used. Indeed it underperforms random filler.

## 14. The precise next milestone

A preregistered study of *donor-use-forcing interface designs*: architectures
or permitted self-supervised objectives that remove or bottleneck the direct
seam-to-seam information path so the child cannot succeed while treating the
donor as a passenger — evaluated first on development data against the same
random-sector twins and bypass ablations, with promotion gates frozen in
advance. Until a candidate beats its random twin *and* loses performance
when the donor is removed, no held-out confirmation should be run and no
inheritance should be claimed.

## Plain-English summary

We built better translators between the two model "languages," and for the
first time the combined child kept the host's language ability within the
agreed limit. But careful detective work — removing the transplanted part,
and swapping it for blank random parts — showed the child's skill does not
come from the transplant at all. The inserted chapters are present and
intact, but the book reads the same (or slightly better) without them. The
honest result is therefore the same category as Campaign 2, now proven much
more rigorously: the surgery works mechanically, but useful inheritance has
not yet been achieved.
