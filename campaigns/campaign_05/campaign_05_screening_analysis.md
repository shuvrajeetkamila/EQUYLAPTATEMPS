# Campaign 5 — Screening Analysis

Benefit vs matched random twin (positive = pretrained better), seed 42 unless noted:

| Sector | Width/position | Benefit | Macro (pretrained) | Donor relative update |
|---|---|---|---|---|
| [20,21) | 1 late | +0.5% | 8.0971 | 0.048 |
| [20,22) | 2 late (C4 ref) | +2.8% | 7.7873 | 0.076 |
| [20,24) | 4 late | −0.7% | 7.8621 | 0.128 |
| [20,26) | 6 late | +4.5% → **−2.1% at seed 123** | 7.7882 | 0.184 |
| [2,6) | 4 early | −1.6% | 7.9135 | 0.211 |
| [12,16) | 4 middle | −0.5% | 7.8464 | 0.099 |

Findings: (1) donor transformation magnitude grows with width, task benefit does
not; (2) no position helps; (3) the largest benefit reversed across seeds, so the
screen's apparent winner is consistent with best-of-six noise; (4) every
candidate fails Gate D (language 1.81–1.87x host). Conclusion: do not expand
budgets or promote; classify and stop this line.
