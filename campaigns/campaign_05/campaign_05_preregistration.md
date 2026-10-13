# Campaign 5 — Preregistration (frozen before training)

Frozen 2026-10-13, before any Campaign 5 real-parent run.

## Question and hypotheses

Can changing the placement/width of the pretrained Qwen sector produce useful
inheritance under Campaign 4's mandatory-use (bottleneck-128 + donor-delta)
architecture? H1: some width/position yields a larger pretrained-specific
benefit at acceptable performance. Competing: wider sectors hurt; position
does not matter; the C4 benefit is a small-sample artifact; the boundary
itself is unsuitable.

## Design (single variable: donor sector; everything else as Campaign 4)

Architecture: bottleneck-128 + donor-delta, host partition [11,13), covariance
bottleneck initialization, 32 AdamW steps, lr 3e-4, seed-42 data protocol
(32+32 train, 16+16 development), CPU float32. Primary metric: development
macro token-NLL (host tokenizer).

- Width screen (late position): [20,21) w1, [20,24) w4, [20,26) w6.
  Reference w2 [20,22) is Campaign 4's verified record (not rerun).
- Position screen (width 4): early [2,6), middle [12,16); late = w4 above.
- Every candidate gets a matched random-sector twin (sector seed 2024,
  identical everything else).

## Frozen gates (identical to Campaign 4)

A integrity/reconstruction; B pretrained macro <= 0.98 x twin macro;
C full macro <= 0.98 x both donor-removed and donor-neutralized macro;
D language <= 1.50 x host AND macro <= 5.7396 (Campaign 3 dense control,
identical protocol); E seed-123 replication of Gate B direction for any
candidate passing A-D. Screening across multiple sectors is exploratory:
no candidate is promoted without Gate E, which controls the
multiple-comparison risk. Held-out confirmation only after A-E;
otherwise NOT_RUN_NO_PROMOTION. Thresholds will not be relaxed post hoc.
