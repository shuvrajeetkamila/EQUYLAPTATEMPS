# Campaign 3 — Failure Analysis

## What was asked

Can a causal, sequence-aware interface let complete pretrained Qwen blocks
operate *usefully* inside a Mamba-host child, beyond Campaign 2's
position-independent dense bridges?

## What the evidence shows (development data only; no TEST selection)

1. **Sequence-aware bridges do help the child — modestly.** All three
   (convolution, attention, GRU) beat the retrained dense-covariance control
   on macro development NLL (best: attention 5.5409 vs dense 5.7396), and all
   pass the historical language-retention gate for the first time
   (ratios 1.416–1.444 vs the 1.50 ceiling; Campaign 2's best was 1.539).
2. **But the donor sector is not doing the work.** Removing the two Qwen
   blocks entirely (`bypass_graft`) changes macro NLL by less than 0.2% for
   every candidate — in several cases the child *without* the donor is
   fractionally better. Zeroing the sector output is catastrophic, so a real
   signal flows through the seam-to-seam path; the donor's own transformation
   of that signal is simply not needed.
3. **The sector is a near-identity passenger.** Inside the trained child the
   donor changes its input by a relative norm of only ≈0.09 (random twin:
   ≈0.017). The bridges learn representations the sector passes through
   almost untouched.
4. **Pretrained content is worse than random content.** At identical
   interface, data, seed and budget, the random-sector twin scores macro
   4.9024 against the pretrained child's 5.5409. Whatever the pretrained
   blocks compute in this foreign residual stream actively hurts relative to
   freshly initialized blocks. (Long-budget twin: see results file.)
5. **More training helps the child, not the inheritance.** Tripling the
   budget improves the attention child to macro 4.3551 (language ratio
   1.269), yet bypass remains identical (4.3550). The gains live in the new
   seams and the host pathway, not in donor usage.
6. **Scale was never the whole story.** Native donor input RMS (29.6) is
   dominated by a first-position spike (87.8 vs ≈4.3 steady-state), so
   Campaign 2's single 36.4× scalar gain overshot most positions. Sequence
   mixers can express position-dependent behavior, which plausibly explains
   their small gain — but matching magnitudes never created semantic
   alignment, exactly as the preregistration warned.

## Interpretation in plain English

Imagine inserting two chapters of a French novel into an English book and
hiring a translator for the joins. The translators (bridges) learned to write
such smooth joins that the French chapters could be removed without the
reader noticing — and a version with blank placeholder chapters actually read
slightly better. The inserted chapters are present, intact, and executed,
but the book's quality does not come from them. That is structural surgery
without useful inheritance.

## What remains genuinely open

- Whether *any* permitted interface objective can force productive use of a
  frozen foreign sector (rather than an easier seam-to-seam shortcut), e.g.
  by constraining the direct seam-to-seam information path by design.
- Whether different sector boundaries or widths change the picture.
- Whether the conclusion survives GPU-scale training budgets.

These are new research questions, not promotions of the current candidates.
No candidate passed the frozen gates; held-out confirmation was not run.
