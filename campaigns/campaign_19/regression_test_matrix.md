# Campaign 19 — Regression test matrix (preparation)

| Area | Test |
|---|---|
| Foreign child execution (cross-width) | test_campaign19_foreign_graft::test_foreign_child_executes_and_freezes_donor |
| Donor frozen / trainable inventory | same test + assert_trainable_contract |
| Unrelated blocks untouched; donor immutable | test_..._unrelated_recipient_blocks_untouched_and_donor_parent_immutable |
| Length-changing graft rejected | test_length_changing_graft_rejected |
| Scramble deterministic, multiset-preserving, non-mutating | test_scramble_deterministic_multiset_preserving_and_nonmutating |
| Random control deterministic, shape-preserving | test_randomize_blocks_deterministic_and_shape_preserving |
| Ablation path executes | test_bypass_ablation_executes_without_donor |
| Prior campaigns | full suite unchanged (see test_report.txt) |
