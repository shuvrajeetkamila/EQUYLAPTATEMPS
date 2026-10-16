# Campaign 8 Screening Analysis (raw-record derived; publisher: `scripts/publish_campaign08.py`)

Host language dev NLL: 4.5595. Macro = mean(language, math) dev NLL; lower is better.

| Condition | Steps | Seed | Pretrained | Twin | Benefit | G-B | G-C |
|---|---|---|---|---|---|---|---|
| original | 32 | 42 | 8.5719 | 7.5349 | −13.76% | FAIL | FAIL |
| original | 32 | 123 | 9.4023 | 7.8900 | −19.17% | FAIL | FAIL |
| calibrated (noscale) | 32 | 42 | 11.6572 | 5.3443 | −118.12% | FAIL | FAIL |
| calibrated (noscale) | 32 | 123 | 8.2784 | 5.3157 | −55.74% | FAIL | FAIL |
| original | 96 | 42 | 7.7585 | 6.0743 | −27.73% | FAIL | FAIL |
| calibrated (noscale) | 96 | 42 | 8.0651 | 4.0586 | −98.72% | FAIL | FAIL |

## Findings

1. **H1 rejected.** Calibration never helped the pretrained child; at 32 steps
   it made it worse (8.57→11.66 seed 42; bypass 8.48 < intact, i.e. the trained
   child was better off *without* the donor pathway). The same calibration
   dramatically improved the random twin (7.53→5.34; 7.89→5.32). Calibration is
   a general conditioning aid that favors near-identity random sectors — the
   exact confound the 2×2 design exists to detect.
2. **H2 rejected.** Tripling the budget improved every child (calibrated twin
   reached 4.0586, language 1.211× host — the best child in this pairing) but
   the pretrained deficit *grew* in relative terms (−27.7% original, −98.7%
   calibrated). More training does not close the gap.
3. **Donor dependence persists without benefit.** Pretrained graft relative
   updates remain ≈19–58 (twins ≈0.19–0.47); ablations move pretrained macros
   substantially. Dependence ≠ inheritance, for the eighth time.
4. No gate passed anywhere; promotion impossible; held-out
   `NOT_RUN_NO_PROMOTION`. **Classification: Outcome C** — the amended
   experiment executed fully and validly, so this is a scientific negative,
   not Outcome D (the frozen method's non-execution is a documented
   portability finding, remedied by Amendment 01 before any amended training).
