# Campaign 11 Structure-Destroyed Control — Design (frozen)

## Construction (`controls.shuffled_sector_control`)

1. Build the child exactly as the intact condition does
   (`child_from_parents`, same recipe/interfaces get installed later).
2. **Deep-copy** the graft blocks (the pinned donor parent is never
   mutated; its fingerprint is verified by the runner as usual).
3. For each of the 42 eligible 2D tensors (7 per block × blocks [24,30)),
   flatten and apply a deterministic permutation drawn from a single
   `torch.Generator` seeded **`20241014`**, iterating tensors in module
   order; reshape back. This preserves each tensor's exact value multiset.
4. The 12 RMSNorm gain vectors are left intact (documented exclusion).
5. Record per-tensor: name, shape, SHA-256 before/after, and a runtime
   multiset-equality check (sorted values identical). The sector's
   post-shuffle fingerprint therefore differs from the donor's while every
   per-tensor statistic is provably preserved.
6. Blocks remain frozen (`requires_grad=False`); only the usual two seams
   train. The control executes the identical Llama computation.

## Validity criteria (checked from run records, frozen)

The control is **valid** iff: (i) all multiset checks pass; (ii) trained
graft relative update is finite and ≥ 0.05 (not silent; Campaign 9's
sleeping twin was ≈0.001, Campaign 10's awake ≈0.60, intact ≈0.48–1.34);
(iii) bypass and zero ablations produce finite, ordered effects
(zero ≥ intact is not required, but non-finite = invalid); (iv) parent
fingerprints unchanged. If invalid: the comparison is reported as
inconclusive for H11 (Finding C / Outcome D pathway) unless a dated
amendment defines a label-free repair before confirmatory runs.
No task outcome is ever used to select or tune the permutation (one seed,
fixed in advance).
