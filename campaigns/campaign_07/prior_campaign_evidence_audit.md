# Campaign 7 Prior-Evidence Audit (2026-10-14)

Scope: what Campaigns 1–6 actually established, separated by evidence strength.
Campaign 7 must not re-litigate reproduced facts nor inherit unverified ones.

## Reproduced in this repository (code + raw records + checksums present)

| Claim | Evidence | Strength |
|---|---|---|
| Mamba→Qwen2.5-Math dense child trains; dev macro ≈5.74 (C2/C3 era) | `campaigns/campaign_02/`, run `38354043771` | Reproduced |
| Sequence interfaces stay bypass-equivalent; random twin beats pretrained at 32 and 96 steps | C3 runs `38367618192`, `38371104410`, `38375777144` | Reproduced |
| Mandatory pathway (bottleneck-128 + donor-delta): small twin benefit at seed 42 (+2.8%), direction preserved at seed 123 (+1.7%); usefulness gates failed | C4 runs `38393669406`, `38397294665`, `38401090731` | Reproduced |
| Sector placement/width sweep: benefit does not scale with donor relative update; the one Gate-B pass (6-block late, seed 42) **reversed at seed 123** | C5 runs `38409870806`, `38414175865`, `38418551528`, `38422937098` | Reproduced |
| Mandatory pathway is intrinsically lossy: widening to 256 or tripling budget (random sectors) stays ≥7.07 macro vs 4.01 unconstrained reference | C6 run `38428594451` | Reproduced |
| Held-out never opened in any campaign (`NOT_RUN_NO_PROMOTION` throughout) | Campaign registries/reports | Reproduced |

## Reported but not independently re-verified here

- Historical held-out BPB (Mamba 1.792145, Qwen 2.343091, child 3.279577, random
  3.043678) and the 12/16 vs 0/16 math diagnostic come from early pilot-era runs
  (`38293921543`, `38301210044`, `38310648576`). Records exist in-repo; Campaign 7
  does not depend on them and does not re-report them as new evidence.

## Unverifiable / explicitly not claimed

- No campaign has ever shown useful pretrained inheritance (Outcome A). Campaign
  4's Outcome B is a small, interface-conditional twin benefit; Campaign 5/6
  showed it does not scale or replicate at the best sector — treat C4's effect as
  the favorable tail of a noisy distribution, not an established phenomenon.
- Nothing in Campaigns 1–6 licenses conclusions about **other pairings**; that
  generalization failure is precisely what Campaign 7 tests.

## Consequences frozen into Campaign 7's design

1. Donor dependence (removal catastrophe) is a diagnostic, never a success gate.
2. The matched random twin is the primary comparator; multiple seeds required
   before any benefit claim (C5 reversal lesson).
3. Mandatory-pathway designs are deprioritized (C6 capacity falsification); the
   primary interface is the simple full-size dense seam, and donor use is judged
   causally by twin + bypass/zero ablations rather than by construction.
