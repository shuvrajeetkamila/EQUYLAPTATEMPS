# HETEROGENEOUS AI MODEL SURGERY — Campaign 8 Final Report

**Campaign 8: Host-Statistics Interface Calibration, Training-Signal
Discovery, and Reproducible Capability Transfer** · Status: **COMPLETE** ·
Development-only; held-out **NOT_RUN_NO_PROMOTION** · Branch
`arena/97efdf4a-equylaptatemps` (session-fixed; the brief's suggested branch
name could not be created), PR #1.

## Final decision: Outcome C — no demonstrated useful inheritance

Campaign 8 kept Campaign 7's pairing exactly (Mamba-130M host, SmolLM-135M
blocks [24,30), dense seams, frozen protocol and gates) and changed only the
interface initialization: host-statistics calibration
(`covariance_orthogonal`, amended to `covariance_orthogonal_noscale` — see
below), in a 2×2 design (pretrained/random × original/calibrated) × two
budgets (32/96 steps) × two seeds.

| Condition | Steps | Seed | Pretrained | Twin | Benefit |
|---|---|---|---|---|---|
| original | 32 | 42 | 8.5719 | 7.5349 | −13.76% |
| original | 32 | 123 | 9.4023 | 7.8900 | −19.17% |
| calibrated | 32 | 42 | 11.6572 | 5.3443 | −118.12% |
| calibrated | 32 | 123 | 8.2784 | 5.3157 | −55.74% |
| original | 96 | 42 | 7.7585 | 6.0743 | −27.73% |
| calibrated | 96 | 42 | 8.0651 | 4.0586 | −98.72% |

**Calibration worked — for the random twin.** It improved the twin at every
budget and seed (best: 4.0586, language 1.211× host) and never helped the
pretrained child (at 32 steps it actively hurt: bypassing the donor beat
using it, 8.48 < 11.66). The pretrained sector lost all six paired
comparisons. H1 (calibration unlocks donor value) and H2 (more training
closes the gap) are both rejected for this pairing.

## Answers to the campaign questions

1. **Did calibration improve activation compatibility?** Diagnostically it
   changed the interface as designed (whitening + orthogonal pair; Stage B
   had measured a ≈186× RMS and covariance mismatch), but "compatibility" in
   any useful sense did not follow for the pretrained sector.
2. **Did it improve the pretrained child vs its original interface?** No —
   worse at 32 steps (11.66 vs 8.57), still worse at 96 (8.07 vs 7.76).
3. **Vs the calibrated random twin?** Catastrophically worse (−118%, −99%).
4. **Did longer training help both equally?** It helped both, but the twin
   more; the relative deficit grew. Budget is not the missing ingredient.
5. **Replication across seeds?** The negative replicated (seeds 42, 123).
6. **Donor ablation?** Pretrained children remain donor-dependent (relative
   update ≈19–58; large bypass/zero shifts) — dependence without benefit,
   as in every prior campaign.
7. **Usefulness criteria?** Never met by any pretrained child (best language
   ratio 1.76× vs 1.50 ceiling). Ironically, the calibrated *twin* at 96
   steps (1.211×, macro 4.06) would have passed G-C — proof the child
   architecture can perform when the sector is easy to absorb.
8. **Held-out justified?** No. `NOT_RUN_NO_PROMOTION`, untouched.
9. **Compute cost:** ≈1.7 CPU runner-hours across five jobs (61 s failed
   frozen screen; 772–2,407 s per stage job). No GPU used.
10. **What failed?** The originally frozen `covariance_orthogonal` could not
    execute — its Qwen-era safety guard rejected SmolLM's genuine 186×
    native/input RMS ratio. Preserved, documented, and remedied by dated
    Amendment 01 (`covariance_orthogonal_noscale`: out-of-range gains
    skipped/recorded; RMSNorm cancels pure input scale) **before** any
    amended training. All amended runs then completed with integrity intact.
11. **Best next experiment:** donor-native preconditioning — give the sector
    its own family's input geometry instead of asking trained seams to
    invent it (full memo: `campaigns/campaign_08/decision_memo.md`). Not
    launched; the owner's decision.

## Methodological note

Had Campaign 8 tested only "calibrated pretrained vs uncalibrated
pretrained," several budgets would have suggested progress narratives. The
matched calibrated twin exposes the truth: calibration is a general
conditioner that prefers near-identity sectors. This is why the programme
insists on twins — and why this negative is trustworthy.

## Verification and publication

Preregistration frozen at `3844b6a` before training; Amendments 01–02 filed
before the runs they govern. All numbers regenerated from SHA-256-verified
raw records by `scripts/publish_campaign08.py` and traced by
`tests/test_campaign08_results.py`; full record in `campaigns/campaign_08/`
(audit, statistics, registries, integrity/resource/failure/test/recovery
docs, decision memo). Final remote SHA, commits, and CI status: PR #1 and
the closing summary of this campaign's publication commit.

## Plain-English closing summary

We tried giving the borrowed SmolLM brain-slice a carefully measured
"adapter fitting," tuned using statistics from the host's own activity —
and, to keep ourselves honest, we gave the exact same fitting to the
random-blocks twin. The fitting turned out to be genuinely good… for the
random twin, which became the best-performing child this pairing has ever
produced. The real pretrained slice, fitted identically, still lost —
everywhere, at every budget, at both seeds. So the problem is not that our
connectors are uncalibrated, nor that training is too short. Something about
a real transformer's inner workings is simply hard for a foreign host to
use through connectors of this kind, while "quiet" random blocks are easy.
That is **Outcome C** again — but a sharper, more informative negative than
before, because it rules out the two most plausible excuses (calibration
and budget) in one campaign.
