# Campaign 19 — Implementation design (preparation)

## Built in this preparation (offline-tested, no real weights)

`src/equylaptatemps/native_placement.py` gains:
- `build_foreign_graft_child(...)`: replaces exactly one declared
  recipient span with an equal-length foreign sector at native
  depth; graft executes through `StackSegment` with the *donor*
  config (native Mamba call convention); seeded `DenseSeam`
  connectors installed as the only trainable components; returns a
  placement manifest (recipient blocks untouched/replaced, donor
  blocks inserted, fingerprints, widths). Length-changing grafts
  and out-of-depth spans raise before assembly.
- `scramble_blocks(blocks, seed)`: Campaign 11 within-tensor
  permutation on deep copies (multiset-preserving, recorded).
- `randomize_blocks(blocks, seed)`: distribution-matched random
  control on deep copies (per-tensor mean/std preserved).

## To be built only after authorization

- `scripts/campaign19_runner.py` (D0 donor validity, A/B gates,
  step-0 scores; then C/D/E/H training via the Campaign 15/16
  segment-training procedure; F/G from trained checkpoints; bundle
  publication per campaign conventions).
- `.github/workflows/campaign19-*.yml` (push-tagged, CPU).
- Publisher deriving the label from frozen decision rules.

## Connector-only-success guards (by construction + tests)

Donor blocks are frozen (`requires_grad_(False)`, contract asserted);
ablation `bypass_graft` removes donor+connectors as one assembly
(limitation recorded); D/E share connectors/budget with C; evaluation
code identical across conditions; no routing/ensembling exists in
`SurgicalChild`.
