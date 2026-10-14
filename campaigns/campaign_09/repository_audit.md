# Campaign 9 Repository & Evidence Audit (Phase 0, 2026-10-14)

## Repository state (verified live, not assumed)

- Branch: `arena/97efdf4a-equylaptatemps` (session-fixed; brief's dedicated
  branch names cannot be created here — same constraint as Campaigns 7–8).
- Local HEAD after restore: `6ef927b1ad69fd98fb1f2947a6d84c676ee6009b`
  ("Campaign 8 complete"), equal to the remote branch head (verified via API).
- PR #1: OPEN, head `6ef927b`, title "Campaigns 1-8 complete — interface
  calibration tested, Outcome C". CI on that commit: success.
- Note: the session-restore artifact recurred at start (local tree at base
  commit `ecc8b0d`, partial files, `.venv` absent). Recovered non-destructively
  from the remote (`fetch` → `reset --mixed` to remote head → `restore`);
  working tree verified clean; nothing was lost or overwritten.
- Sandbox resources: no GPU, ~3.8 GiB RAM, 2+ CPUs, HF unreachable directly —
  real-parent runs execute on GitHub Actions CPU runners via the established
  opt-in workflows, exactly as Campaigns 2–8. Framework: torch 2.5.1,
  transformers 4.48.3 (Actions); local `.venv` rebuilt for tests only.

## Campaign 8 verification from primary evidence (Step 0.3)

All twelve published macros (Campaign 7 stored + Campaign 8 new) were
recomputed from the raw per-run `results.json` records and match
`campaigns/campaign_08/campaign_08_results.json` exactly; outcome **C**,
promotion false, held-out `NOT_RUN_NO_PROMOTION`. The brief's cited values
(8.5719/7.5349, 9.4023/7.8900, twin best 4.0586) are confirmed; no discrepancy.

## Interface anatomy (answers to Step 0.3 questions, from source)

1. **What the interface does:** two `DenseSeam`s — LayerNorm + full Linear
   (768→576 in, 576→768 out) — the only trainable tensors (888,768)
   (`src/equylaptatemps/model.py`).
2. **Entry point:** host prefix output (Mamba blocks [0,11)) → input seam →
   graft (SmolLM blocks [24,30)) → output seam → host suffix [13,24).
3. **Learned vs fixed:** seams learned (or statistically initialized, C8);
   everything else frozen; donor internals bitwise-verified per run.
4. **Donor input expectation:** the residual stream produced by SmolLM blocks
   [0,24) in its native model — measured native RMS ≈196.5, with donor-native
   covariance (available: `measure_initial` stores covariance tensors for all
   streams, incl. `donor_native_input/covariance`).
5. **Donor output:** residual stream after block 29, natively consumed by
   SmolLM's final RMSNorm; in the child it is consumed by the output seam.
6. **Transformations:** per-token RMSNorm at each Llama block entry (cancels
   pure input scale — geometry/direction is the operative mismatch), RoPE
   inside the graft (`StackSegment`), additive causal mask.
7. **Access to donor-native geometry:** the existing interface has *none* —
   Campaign 8 calibration used child-side covariance + scalar RMS only. The
   donor-native covariance tensor is measured but has never been used to
   construct the interface. **That unused measurement is Campaign 9's lever.**
8. **Unknowns:** per-feature native statistics beyond second moments, and any
   semantic (not merely geometric) alignment, cannot be inferred from config
   or covariance alone — the preregistration treats geometry matching as a
   testable intervention, not as established semantic alignment.
